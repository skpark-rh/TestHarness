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

from .config import HarnessConfig
from .testharness import CrossNodeTestHarnessBase, _log
from .skips.nvshmem import XFAIL_WHEN_1GPU_PER_POD, SUITES
from .skips.nccl import MODULES


# status per test: if 0 PASS, 1 SKIP, 2 FAIL. All-reduce MAX so the WORST outcome on any rank governs.
OK, SKIP, FAIL, XFAIL = 0, 1, 2, 3

class CrossMultiProcContinousTH(CrossNodeTestHarnessBase):
    def __init__(
            self,
            module: str,
            ngpus: int,
            nodes: int,
            ref: str = "main",
            repo: str = "pytorch",
            classes: list[str] | None = None,
            timeout: int = 300,
            no_poison: bool = False,
            list_flag: bool = False
        ) -> None:
        super().__init__(module, MultiProcContinuousTest, classes, ngpus, nodes, ref, repo)

        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._world_size = int(os.environ["WORLD_SIZE"])
        self._local_world = int(os.environ.get("LOCAL_WORLD_SIZE", self._world_size))
        self._is_rank0 = self._rank == 0

        self._list_flag = list_flag
        self._no_poison = no_poison
        self._timeout = timeout

        # self._plan = []
        self._poisoned_classes = set()
        self._results = []  # (cname, test, status, detail)
        self._device = None

        if self._only_classes:
            self._unknown = self._only_classes - {n for n, _ in self._classes}
            if self._unknown and self._is_rank0:
                _log(f"WARNING: unkown class name(s) ignored: {', '.join(sorted(self._unknown))}")
        if not self._classes:
            _log(f"no MultiProcContinuousTest classes selected in {os.path.basename(self._path)} -- SKIP")
            sys.exit(0)
        # self._set_plan()
        self._profile_guard()
        self._initialize_pg()
        self._prime_classes()

        _log(f"world_size={self._world_size}  ({self._local_world} GPU/pod x {self._world_size // max(self._local_world, 1)} nodes)    "
             f"backend=nvshmem module={os.path.basename(self._path)}   tests={len(self._plan)}")

    # def _set_plan(self):
    #     """Build the full ordered (class, method) plan -- identical on every rank"""
    #     for cname, cls in self._classes:
    #         for t in self._collect_tests(cls):
    #             self._plan.append((cname, cls, t))

        # if self._list_flag:
        #     if self._is_rank0:
        #         _log(f"# {os.path.basename(self._path)}: {len(self._classes)} class(es), {len(self._plan)} test(s)")
        #         cur = None
        #         for c_name, _, t in self._plan:
        #             if c_name != cur:
        #                 _log(f"\n{c_name}:")
        #                 cur = c_name
        #             _log(f"     {t}")
        #     sys.exit(0)

        # if not self._plan:
        #     _log("no tests matched the selection -- SKIP")
        #     sys.exit(0)

    def _profile_guard(self):
        # NCCL preflight warn
        if "nccl" in self._mod_name:
            prof = MODULES.get(os.path.basename(self._path).removesuffix(".py"))
            if prof and self._is_rank0:
                if self._local_world < prof["min_gpus_per_pod"]:
                    _log(f"WARNING: {self._mod} needs >= {prof['min_gpus_per_pod']} GPU/pod to RUN "
                         f"(TEST_MULTIGPU); got {self._local_world}. Most tests will self skip. Use --bucket 4gpu.")
                if self._world_size < prof["min_world"]:
                    _log(f"WARNING: {self._mod} needs world >={prof['min_world']}; got {self._world_size}. "
                         f"Some tests will skip_if_lt_x_gpu.")
        return

    def _initialize_pg(self):
        self._device = torch.device("cuda", self._local_rank)
        torch.cuda.set_device(self._device)
        os.environ["LOCLAL_RANK"] = str(self._local_rank) # some init paths read this
        dist.init_process_group(
            backend="nccl", # this is just a place holder to initialize.
            world_size=self._world_size,
            rank=self._rank,
            device_id=self._device,
            timeout=datetime.timedelta(seconds=self._timeout)
        )
        self._pg = c10d._get_default_group()

    def _prime_classes(self):
        """
        Prime every class the way upstream harness expects its workers to be primed.
            - cls.rank / cls.world_size = the true global rank/world (collectives depend on it)
            - cls.pg = the shared default group
            - device property -> cuda:LOCAL_RANK (the one cross-node fix; leaves self.rank global)
            - cls.rank_to_GPU -> {r: [LOCAL_RANK]} Every rank maps to this pod's local device.  A process
              only builds tensors on its own device, so mapping all keys to local_rank is the move and
              this sidesteps `init_multigpu_helper`'s single-host math.
            - class timeout honored
        """
        local_dev_prop = property(lambda self: torch.device("cuda", self._local_rank))
        rank_to_gpu_prop = property(lambda self: {r: [self._local_rank] for r in range(self._world_size)})
        for _, cls in self._classes:
            cls.rank = self._rank
            cls.world_size = self._world_size
            cls.pg = self._pg
            cls.device = local_dev_prop
            if "nccl" in self._mod_name:
                cls.rank_to_GPU = rank_to_gpu_prop
            cls.timeout = datetime.timedelta(seconds=self._timeout)

        return

    def run(self) -> int:
        for cname, cls, test in self._plan:
            # In case of poison pill tests, skip the rest of the class as the PG may be corrupted.
            # Keep other ranks in lockstep.
            if cname in self._poisoned_classes:
                self._results.append((cname, test, SKIP, "poisoned: earlier test in class failed."))
                continue

            dist.barrier()
            local_status, detail = self._run_one(cls, test)
            status = self._reduce_status(local_status, self._device)
            self._results.append((cname, test, status, detail))

            if status == FAIL:
                if not self._no_poison:
                    self._poisoned_classes.add(cname)
                if self._is_rank0:
                    _log(f"[FAIL] {cname}.{test}")
                    if local_status == FAIL and detail:
                        _log(detail.rstrip())
            elif self._is_rank0:
                _log(f"[{'PASS' if status == OK else 'SKIP'}] {cname}.{test}"
                     + (f"  ({detail})" if status == SKIP and detail else ""))

        # --- summarize on rank 0 ---
        npass = sum(1 for r in self._results if r[2] == OK)
        nskip = sum(1 for r in self._results if r[2] == SKIP)
        nfail = sum(1 for r in self._results if r[2] == FAIL)
        nxfail = sum(1 for r in self._results if r[2] == XFAIL)
        if self._is_rank0:
            _log("\n" + "=" * 60)
            _log(f"CROSS-NODE UPSTREAM RUN  ({os.path.basename(self._path)})")
            _log(f"  {npass} passed, {nfail} failed, {nxfail} xfailed, {nskip} skipped  "
                 f"(of {len(self._results)})")
            if nfail:
                _log("  FAILED:")
                for cname, test, s, _ in self._results:
                    if s == FAIL:
                        _log(f"    {cname}.{test}")
            if nxfail:
                _log("  XFAILED (expected on this topology):")
                for cname, test, s, _ in self._results:
                    if s == XFAIL:
                        _log(f"    {cname}.{test}")
            _log("ALL DONE" if nfail == 0 else "FAILURES ABOVE")

        sys.stdout.flush()
        sys.stderr.flush()
        return 1 if nfail else 0

    @staticmethod
    def _cleanup():
        """Tear down distributed state. Safe to call multiple times / before init."""
        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier()
            except Exception:  # noqa: BLE001
                pass
            dist.destroy_process_group()

        return

    @staticmethod
    def _reduce_status(local_status: int, dev: torch.device) -> int:
        t = torch.tensor([local_status], device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    @staticmethod
    def _run_one(cls, test) -> tuple[int, str]:
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
    # argparser.add_argument("--suite", choices=sorted(SUITES),
    #                        help="Choices: (internode, mixed, intranode, subgroup). This selects the tests" \
    #                        "by mechanism and enforces matching GPU placement.")
    # argparser.add_argument("--module", default="test_nvshmem", help="Upstream test module to run that is under test/distributed (e.g. test_nvshmem)")
    # argparser.add_argument("--test-file", default=None, help="Absolute path to the upstream test*.py. (Overrides --module)")
    # argparser.add_argument("--pytorch-src", default=os.environ.get("PYTORCH_SRC"),
    #                        help="Path to a local pytorch checkout to pull test/distributed/<module>.py from. "
    #                             "(default: $PYTORCH_SRC; if unset, fetches from upstream GitHub)")
    # argparser.add_argument("--ref", default="main",
    #                        help="Upstream git ref (branch/tag/SHA) to fetch the test file from when no local "
    #                             "source is found. Default: main (latest).")
    # argparser.add_argument("--classes", default=None, help="Comma-separated list of test classes to run. (default: all in module)")
    # argparser.add_argument("--tests", default=None, help="Common list of substrings which only run matching test methods.")
    # argparser.add_argument("--exclude", default=None, help="Comma-separated list of substrings that skip matching test methods.")
    # argparser.add_argument("--timeout", type=int, default=300, help="Process group / timeout in seconds. Default: 300s")
    # argparser.add_argument("--no-poison", action="store_true", help="Do not poison the process group on test failure. Default: False")
    # argparser.add_argument("--list", action="store_true", help="List selected classes/tests and exit (rank 0 only).")
    argparser.add_argument("--b64-cfg", type=str, help="Base 64 encoded json string of the harness config.")
    args = argparser.parse_args()

    cfg = HarnessConfig.from_b64(args.b64_cfg)
    cn_mpcth = CrossMultiProcContinousTH(
        module=cfg.module, ngpus=cfg.ngpus, nodes=cfg.nodes,
        ref=cfg.ref, repo=cfg.repo, classes=cfg.classes,
        timeout=cfg.timeout, no_poison=cfg.no_poison,
        list_flag=cfg.list_flag
        )
    code = cn_mpcth.run()
    # Fast exit avoid teardown-time NCCL/atexit noise after results are recorded.
    os._exit(code)

if __name__ == "__main__":
    main()
