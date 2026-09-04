"""
Run unmodified upstream `MultiProcContinuousTest` suites across both nodes on RDMA fabric.

Target upstream tests: test_nvshmem.py, test_symmetric_memory.py

The goal is to wrap PyTorch's `MultiProcContinuousTest` which reuses one persistent process group
accross a class's tests. These tests are single node tests and single node assumptions are replaced.

  1. RENDEVOUS: FileStore + local workers -> torchrun `env://` (1 process == 1 global rank)
  2. DISPATCH: cross-node-incapable queue
        A barrier synched loop where every rank enumerates the same tests and all-reduces pass/fail/skip
        so a rank's failure is seen by all. Note that parameterization is deterministic at import.
  3. DEVICE: `device = cuda:self.rank` (global) is an invalid index cross-node.
        This is monkey-patched to `device = cuda:LOCAL_RANK`, leaving self.rank global (collectives need it).

USAGE: This has to work in tandem with RedHat's submit-jobs.py API. It leverages the required torchrun and the
enviornment variables required to run NVSHMEM tests across nodes.
    ./submit-jobs.py -n <ns> --bucket 2gpu --script <path_to_this_file> --script-args "--module test_nvshmem"

Exit code is 0 iff every selected test passed (or skipped) on every rank.    
"""
import argparse
import datetime
import os
import sys
import traceback
import unittest

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_distributed import MultiProcContinuousTest, TEST_SKIPS

from testharness import CrossNodeTestHarnessBase


def _log(*a):
    print(*a, flush=True)

# Transport/topology suits
# Certain nvshmem test are only valid with specific gpu placement. The tests are grouped accordingly.
#   internode --bucket 2gpu (1 GPU/pod x N nodes; every peer hop = NIC / IBRC-IBGDA)
#   mixed     --bucket 4gpu (2 GPU/pod x N nodes; P2P same-node + RDMA cross-node)
#   intranode intranode-nvlink.yml (2 GPU, signle node; NVLINK load/store)
#   subgroup  subgroup-nvlink.yml (4 GPU, single node)

# nvshmem_ptr() load/store + NVLS multicast. Need this with NVSWITCH capabilities.
NVLINK_ONLY = ["get_remote", "multicast_ptr", "tile_reduce", "multi_root",
               "one_shot", "multimem"] # one_shot + multimem is in test_symmetric_memory
NEEDS_4_LOCAL_GPU = ["subgroup"]

# currently test_get is failing on 2 nodes, 1 GPU/pod.
XFAIL_WHEN_1GPU_PER_POD = {
    # "test_get": (
    #     "symm_mem.get() into a non-symmetric (unregistered) local dst is unsupported cross-node "
    #     "(NVSHMEM getmem needs a registered local MR -> ibrc WC status 4, LOC_PROT_ERR). Passes "
    #     "where the peer is P2P-reachable. Upstream op bug in nvshmem_get_out; see the bug doc."
    # )
    "test_barrier": (
        "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    ),
    "test_put_wait_signal": (
        "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    ),
    "test_nvshmem_get": (
        "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    )
}

# Asymmetric NVLS tile reduce hangs. Nee NVLS hardware. Also need NVSHMEM_MAX_TEAMS raised.
ASYMMETRIC_TILE_REDUCE_HANGS = ["test_tile_reduce", "root_ratio_2"]

SUITES = {
    "internode": {
        "expect": "1gpu-per-node",   # local_world_size == 1 -> every peer is remote
        "topology": "1 GPU/pod x N nodes -- every peer hop crosses the NIC (IBRC/IBGDA)",
        "launch": "submit-job.py --bucket 2gpu",
        "only": None,                # all tests except the excludes below
        "exclude": NVLINK_ONLY + NEEDS_4_LOCAL_GPU,   # test_get NOT excluded -> xfail'd at 1 GPU/pod (see dispatch)
    },
    "mixed": {
        "expect": "multi-gpu-multi-node",   # local_world>=2 and nnodes>=2
        "topology": "2 GPU/pod x N nodes -- P2P for same-node peers + RDMA cross-node",
        "launch": "submit-job.py --bucket 4gpu",
        "only": None,
        "exclude": NVLINK_ONLY + NEEDS_4_LOCAL_GPU,   # peer-ptr would DEADLOCK; NVLS would crash
    },
    "intranode": {
        "expect": "single-node",     # nnodes == 1 -> peers are NVLink-reachable
        "topology": "2+ GPU, SINGLE node -- NVLink/NVSwitch load/store domain",
        "launch": "crossnode/intranode-nvlink.yml (torchrun --standalone --nnodes=1 --nproc_per_node=2)",
        "only": NVLINK_ONLY,         # run ONLY the NVLink-domain ops here
        "exclude": NEEDS_4_LOCAL_GPU + ASYMMETRIC_TILE_REDUCE_HANGS,
    },
    "subgroup": {
        "expect": "single-node-4gpu",   # nnodes == 1 AND local_world >= 4
        "topology": "4 GPU, SINGLE node -- DispatchCombineInSubgroups splits WORLD into "
                    ">=2 subgroups, so it needs >=4 local devices (skip_if_lt_x_gpu(4))",
        "launch": "crossnode/subgroup-nvlink.yml (torchrun --standalone --nnodes=1 --nproc_per_node=4)",
        "only": NEEDS_4_LOCAL_GPU,   # run ONLY the >=4-GPU subgroup test(s)
        "exclude": [],               # nothing else -- this is the one topology that CAN run them
    },
}

# status per test: if 0 PASS, 1 SKIP, 2 FAIL. All-reduce MAX so the WORST outcome on any rank governs.
OK, SKIP, FAIL, XFAIL = 0, 1, 2, 3

class CrossMultiProcContinousTH(CrossNodeTestHarnessBase):
    @staticmethod
    def xfail_reason(test: str, xfail_rules: dict) -> str | None:
        for key, reason in xfail_rules.items():
            if key in test:
                return reason
        return None

    @staticmethod
    def reduce_status(local_status: int, dev: torch.device) -> int:
        t = torch.tensor([local_status], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    @staticmethod
    def run_one(cls, test) -> tuple[int, str]:
        """Run a single upstream test body on this rank. Returns the status and detail."""
        try:
            cls._run_test_given_id(f"{cls.__name__}.{test}")
            return OK, ""
        except unittest.SkipTest as ex:
            return SKIP, str(ex)
        except SystemExit as ex:
            code = getattr(ex, "code", None)
            entry = next((v for v in TEST_SKIPS.values() if v.exit_code == code), None)
            if entry:
                return SKIP, entry.message
            return FAIL, f"SystemExit(code={code})"
        except BaseException: # noqa: B036
            return FAIL, traceback.format_exc()


def main():
    argparser = argparse.ArgumentParser(description="Run upstream MultiProcContinuousTest tests across nodes")
    argparser.add_argument("--suite", choices=sorted(SUITES),
                           help="Choices: (internode, mixed, intranode, subgroup). This selects the tests" \
                           "by mechanism and enforces matching GPU placement.")
    argparser.add_argument("--module", default="test_nvshmem", help="Upstream test module to run that is under test/distributed (e.g. test_nvshmem)")
    argparser.add_argument("--test-file", default=None, help="Absolute path to the upstream test*.py. (Overrides --module)")
    argparser.add_argument("--pytorch-src", default=os.environ.get("PYTORCH_SRC"),
                           help="Path to a local pytorch checkout to pull test/distributed/<module>.py from. "
                                "(default: $PYTORCH_SRC; if unset, fetches from upstream GitHub)")
    argparser.add_argument("--ref", default="main",
                           help="Upstream git ref (branch/tag/SHA) to fetch the test file from when no local "
                                "source is found. Default: main (latest).")
    argparser.add_argument("--classes", default=None, help="Comma-separated list of test classes to run. (default: all in module)")
    argparser.add_argument("--tests", default=None, help="Common list of substrings which only run matching test methods.")
    argparser.add_argument("--exclude", default=None, help="Comma-separated list of substrings that skip matching test methods.")
    argparser.add_argument("--timeout", type=int, default=300, help="Process group / timeout in seconds. Default: 300s")
    argparser.add_argument("--no-poison", action="store_true", help="Do not poison the process group on test failure. Default: False")
    argparser.add_argument("--list", action="store_true", help="List selected classes/tests and exit (rank 0 only).")
    args = argparser.parse_args()

    # --- env from torchrun / entrypoint ---
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    is_rank0 = rank == 0

    crossnode_th = CrossMultiProcContinousTH(args.module, args.test_file, MultiProcContinuousTest, args.pytorch_src, args.ref)
    path = crossnode_th.discover_test_file()
    suite = SUITES[args.suite] if args.suite else None
    only_classes = set(args.classes.split(",")) if args.classes else None

    substrings = (args.tests.split(",") if args.tests 
                  else (list(suite["only"]) if suite and suite["only"] else None))
    excludes = args.exclude.split(",") if args.exclude else []
    if suite:
        excludes = sorted(set(excludes) | set(suite["exclude"]))
    excludes = excludes or None

    crossnode_th.import_test_module()
    classes = crossnode_th.collect_classes(only_classes)
    if only_classes:
        unknown = only_classes - {n for n, _ in classes}
        if unknown and is_rank0:
            _log(f"WARNING: unkown class name(s) ignored: {', '.join(sorted(unknown))}")
    if not classes:
        _log(f"no MultiProcContinuousTest classes selected in {os.path.basename(path)} -- SKIP")
        os._exit(0)

    # Build the full ordered (class, method) plan -- identical on every rank
    plan = [(cname, cls, t) for (cname, cls) in classes
            for t in crossnode_th.collect_tests(cls, substrings, excludes)]

    if args.list:
        if is_rank0:
            _log(f"# {os.path.basename(path)}: {len(classes)} class(es), {len(plan)} test(s)")
            cur = None
            for cname, _, t in plan:
                if cname != cur:
                    _log(f"\n{cname}:")
                    cur = cname
                _log(f"    {t}")
        os._exit(0)

    if not plan:
        _log("no tests matched the selection -- SKIP")
        os._exit(0)

    # --- one persistent PG spanning both nodes (env:// rendezvous via MASTER_ADDR/PORT) ---
    dev = torch.device("cuda", local_rank)
    torch.cuda.set_device(dev)
    os.environ["LOCAL_RANK"] = str(local_rank) # some init paths read this
    dist.init_process_group(
        backend="nccl", # this is just a place holder to initialize.
        world_size=world_size,
        rank=rank,
        device_id=dev,
        timeout=datetime.timedelta(seconds=args.timeout)
    )
    pg = c10d._get_default_group()

    # Prime every class the way upstream harness expects its workers to be primed.
    #   - cls.rank / cls.world_size = the true global rank/world (collectives depend on it)
    #   - cls.pg = the shared default group
    #   - device property -> cuda:LOCAL_RANK (the one cross-node fix; leaves self.rank global)
    #   - class timeout honored
    local_dev_prop = property(lambda self: torch.device("cuda", local_rank))
    for _, cls in classes:
        cls.rank = rank
        cls.world_size = world_size
        cls.pg = pg
        cls.device = local_dev_prop
        cls.timeout = datetime.timedelta(seconds=args.timeout)

    if is_rank0:
        _log(f"world_size={world_size}  ({local_world} GPU/pod x {world_size // max(local_world,1)} nodes)   "
             f"backend=nccl module={os.path.basename(path)}  tests={len(plan)}")

    # --- barrier-synchronised dispatch loop ---
    poisoned_classes: set[str] = set()
    results = [] # (cname, test, status, detail)

    # Tests known-unsupported on this topology is recorded as an expected failure.
    xfail_rules = XFAIL_WHEN_1GPU_PER_POD if local_world == 1 else {}

    for cname, cls, test in plan:
        # In case of posion pill tests, skip the rest of the class as the PG may be corrupted.
        # Keep other ranks in lockstep.
        if cname in poisoned_classes:
            results.append((cname, test, SKIP, "poisoned: earlier test in class failed"))
            continue

        # For expected failures record XFAIL and move on without running the op. Don't execute crossnode
        # as the errors aren't raised cleanly as the async-aborts usually hangs.
        xr = crossnode_th.xfail_reason(test, xfail_rules)
        if xr:
            results.append((cname, test, XFAIL, xr))
            if is_rank0:
                _log(f"[XFAIL] {cname}.{test} (expected failure on this topology)")
                _log(f"    {xr}")
            continue

        dist.barrier()
        local_status, detail = crossnode_th.run_one(cls, test)
        status = crossnode_th.reduce_status(local_status, dev)
        results.append((cname, test, status, detail))

        if status == FAIL:
            if not args.no_poison:
                poisoned_classes.add(cname)
            if is_rank0:
                _log(f"[FAIL] {cname}.{test}")
                if local_status == FAIL and detail:
                    _log(detail.rstrip())
        elif is_rank0:
            _log(f"[{'PASS' if status == OK else 'SKIP'}] {cname}.{test}"
                 + (f"  ({detail})" if status == SKIP and detail else ""))

    # --- summaize on rank 0 ---
    npass = sum(1 for r in results if r[2] == OK)
    nskip = sum(1 for r in results if r[2] == SKIP)
    nfail = sum(1 for r in results if r[2] == FAIL)
    nxfail = sum(1 for r in results if r[2] == XFAIL)
    if is_rank0:
        _log("\n" + "=" * 60)
        _log(f"CROSS-NODE UPSTREAM RUN  ({os.path.basename(path)})")
        _log(f"  {npass} passed, {nfail} failed, {nxfail} xfailed, {nskip} skipped   "
             f"(of {len(results)})")
        if nfail:
            _log("  FAILED:")
            for cname, test, s, _ in results:
                if s == FAIL:
                    _log(f"    {cname}.{test}")
        if nxfail:
            _log("  XFAILED (expeted on this topology):")
            for cname, test, s, _ in results:
                if s == XFAIL:
                    _log(f"    {cname}.{test}")
        _log("ALL DONE" if nfail == 0 else "FAILURES ABOVE")

    sys.stdout.flush()
    sys.stderr.flush()
    # Fast exit avoid teardown-time NCCL/atexit noise after results are recorded.
    os._exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
