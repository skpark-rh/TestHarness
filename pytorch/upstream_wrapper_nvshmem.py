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
import inspect
import os
import sys
import tempfile
import traceback
import unittest
import urllib.error
import urllib.request

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_distributed import MultiProcContinuousTest, TEST_SKIPS
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from importlib.util import spec_from_file_location, module_from_spec


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


UPSTREAM_RAW = "https://raw.githubusercontent.com/pytorch/pytorch/{ref}/test/distributed/{module}.py"

# status per test: if 0 PASS, 1 SKIP, 2 FAIL. All-reduce MAX so the WORST outcome on any rank governs.
OK, SKIP, FAIL, XFAIL = 0, 1, 2, 3

class CrossNodeTestHarness:
    def __init__(self, module: str, test_file: str | None, pytorch_src: str | None = None,
                 ref: str = "main") -> None:
        self._mod_str = module
        self._test_file_pth = test_file
        self._pytorch_src = pytorch_src
        self._ref = ref
        self._path = None
        self._mod = None

    def discover_test_file(self) -> str:
        """Locate an upstream test file: explicit file -> local clone -> upstream fetch.

        The installed `torch` is usually a wheel (torch.__file__ lives under site-packages),
        which ships no `test/` tree, so we can't rely on it. Prefer a local pytorch checkout
        and only download from GitHub as a last resort.
        """
        rel = os.path.join("test", "distributed", f"{self._mod_str}.py")

        # 1. explicit file wins, no discovery.
        if self._test_file_pth:
            if not os.path.isfile(self._test_file_pth):
                sys.exit(f"--test-file not found: {self._test_file_pth}")
            self._path = self._test_file_pth
            return self._path

        # 2. an explicit local pytorch checkout (--pytorch-src / $PYTORCH_SRC).
        if self._pytorch_src:
            cand = os.path.join(self._pytorch_src, rel)
            if not os.path.isfile(cand):
                sys.exit(f"--pytorch-src given but test file not found:\n  {cand}")
            self._path = cand
            return self._path

        # 3. an editable/source install: torch.__file__ == <src>/torch/__init__.py, so
        #    two dirnames land on a repo root that also holds test/. (No-op for a wheel.)
        src_root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        cand = os.path.join(src_root, rel)
        if os.path.isfile(cand):
            self._path = cand
            return self._path

        # 4. fall back to downloading the latest matching file straight from upstream GitHub.
        self._path = self._fetch_upstream()
        return self._path

    def _fetch_upstream(self) -> str:
        """Download the latest test/distributed/<module>.py from pytorch/pytorch to a temp file.

        Defaults to the `main` branch (the newest test source, not the installed release,
        whose tag would pull a stale copy). Override with --ref for a specific tag/branch/SHA.
        """
        url = UPSTREAM_RAW.format(ref=self._ref, module=self._mod_str)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            sys.exit(
                f"could not fetch upstream test file '{self._mod_str}.py' from "
                f"pytorch/pytorch@{self._ref}: HTTP {e.code} ({url})\n"
                f"  check --module/--ref, or pass --pytorch-src <clone> / --test-file <abs_path>."
            )
        except urllib.error.URLError as e:
            sys.exit(f"could not reach GitHub to fetch '{self._mod_str}.py' ({url}): {e}")
        fd, tmp = tempfile.mkstemp(prefix=f"{self._mod_str}_", suffix=".py")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        _log(f"fetched upstream {self._mod_str}.py from pytorch/pytorch@{self._ref}")
        return tmp

    def import_test_module(self):
        """Import an upstream test module by file path, with test/ dirs on sys.path (as pytest would)."""
        dist_dir = os.path.dirname(os.path.abspath(self._path))
        test_dir = os.path.dirname(dist_dir)
        for p in (dist_dir, test_dir):
            if p not in sys.path:
                sys.path.insert(0, p)
        modname = os.path.splitext(os.path.basename(self._path))[0]
        spec = spec_from_file_location(modname, self._path)
        mod = module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        self._mod = mod

    def collect_classes(self, only_classes: set[str] | None):
        """Ordered list of MultiProcContinuousTest subclasses defined in `mod` (optionally filtered)."""
        out = []
        for name, obj in vars(self._mod).items():
            if (inspect.isclass(obj) and issubclass(obj, MultiProcContinuousTest)
                    and obj is not MultiProcContinuousTest and obj.__module__ == self._mod.__name__):
                if only_classes and name not in only_classes:
                    continue
                out.append((name, obj))
        out.sort(key=lambda kv: kv[0])
        return out

    def collect_tests(self, class_obj, substrs, excludes):
        """Deterministic sorted test-method names for a class. Ensure the parameterized methods are expanded."""
        # Expand @parametrize template only for classes that weren't already decorated with
        # @instantiate_parametrized_tests at import. `instantiate_parametrized_tests` is not idempotent.
        def _has_raw_template(c):
            for n in dir(c):
                a = getattr(c, n, None)
                if hasattr(a, "parametrize_fn") and getattr(a, "__name__", None) == n:
                    return True
            return False
        if _has_raw_template(class_obj):
            instantiate_parametrized_tests(class_obj)
        names = sorted(n for n in dir(class_obj) if n.startswith("test"))
        if substrs:
            names = [n for n in names if any(s in n for s in substrs)]
        if excludes:
            names = [n for n in names if not any(s in n for s in excludes)]
        return names

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

    crossnode_th = CrossNodeTestHarness(args.module, args.test_file, args.pytorch_src, args.ref)
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
