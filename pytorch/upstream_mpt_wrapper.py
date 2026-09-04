"""
Run unmodified upstream MultiProcessTestCase (mpt) across nodes.

The mpt self spawns a local world (fork + `FileStore(self.filename, world)` on a per-rank tempfile).
So env:// can't make that world span two pods.

This file wraps mpt, via three facts about mpt.

1.  `join_or_run` wraps every test method. When `self.rank != MAIN_PROCESS_RANK` the wrapper runs the real body in this process (no spawn).  
Torchrun's "one process per global rank" is the world. Set self.rank to the global rank and call the wrapped method.
2. Every body rendezvous with `c10d.FileStore(self.filename, self.world_size)`. 
Monkeypatch FileStore -> a PrefixStore over one persistent cross-node TCPStore (a fresh prefix per test).  
This forms a PG, `init_process_group(store=..., rank=self.rank, world_size=self.world_size)`, that span both pods and every collective crosses the RoCE/RDMA fabric
3. Each class hardcodes its own world_size. This is not overrided. 
Only run tests that matches the given world_size to the declared world_size in the tests.  
Tests will be grouped accordingly.

DEVICE PLACEMENT: test bodies place tensors by global rank -- `.cuda(self.rank)`, `.to(self.rank)`, `set_device(self.rank)`,
etc... This is unpatchable from the python side. So a rank's GPU ordinal must physically exist on its pod. Run this under
"decoupled" training runtime manifests. Each pod calims `world` GPUs but runs only `world/2` procs, so cuda:self.rank is
valid on every pod and every test runs unmodified with no device monkeypatching. (A little wasteful but it gets the job done.)
Ranks are still split across 2 nodes, so every collective crosses the RDMA fabric.

USAGE...asdf
"""
import argparse
import datetime
import inspect
import functools
import os
import subprocess
import sys
import threading
import traceback
import unittest
from importlib.util import spec_from_file_location, module_from_spec

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS

from testharness import CrossNodeTestHarnessBase


OK, SKIP, FAIL = 0, 1, 2


def _log(*a):
    print(*a, flush=True)


class StoreCoord:
    """
    Barrier + worst-status reduction across all physical ranks using only the TCPStore -- no PG,
    so it never touches the default-PG slot the tests themselves init/destroy.
    """
    def __init__(self, store, rank, world, timeout_s):
        self.store, self.rank, self.world = store, rank, world
        self.timeout = datetime.timedelta(seconds=timeout_s)

    def barrier(self, tag):
        done = f"b/{tag}/done"
        if self.store.add(f"b/{tag}/n", 1) == self.world:
            self.store.set(done, "1")
        self.store.wait([done], self.timeout)

    def max_status(self, tag, local):
        """Return max(local status over all ranks): FAIL(2) > SKIP(1) > PASS(0)."""
        self.store.set(f"s/{tag}/{self.rank}", str(int(local)))
        self.barrier(tag)
        return max(int(self.store.get(f"s/{tag}/{r}")) for r in range(self.world))


class CrossMultiProcessTH(CrossNodeTestHarnessBase):
    def __init__(
            self,
            module,
            test_file,
            base,
            no_guard,
            test_timeout,
            pg_timeout,
            classes,
            excludes,
            tests,
            pytorch_src = None,
            ref = "main"
        ):
        super().__init__(module, test_file, base, pytorch_src, ref)
        # Patch FileStore -> PrefixStore(<per-test key>, base_store). SPMD: every rank makes the same FileStore calls in the same order.
        self._cur = {"key": "init", "n": 0}

        self._no_guard = no_guard
        self._test_timeout = test_timeout
        self._pg_timeout = pg_timeout
        self._classes = classes
        self._excludes = excludes
        self._tests = tests

        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._world = int(os.environ["WORLD_SIZE"])
        self._local_world = int(os.environ.get("LOCAL_WORLD_SIZE", self._world))
        self._master_addr = os.environ["MASTER_ADDR"]
        self._master_port = int(os.environ["MASTER_PORT"])
        self._is_rank0 = self._rank == 0

    def run_one(self, cls, test, tag):
        """Run one upstream test body on this rank.  Returns (status, detail)."""
        env_snapshot = dict(os.environ)
        inst = cls(test)
        try:
            inst.setUp()
        except unittest.SkipTest as ex:
            return SKIP, str(ex)
        except SystemExit as ex:
            code = getattr(ex, "code", None)
            entry = next((v for v in TEST_SKIPS.values() if v.exit_code == code), None)
            return (SKIP, entry.message) if entry else (FAIL, f"setUp SystemExit(code={code})")
        except Exception:
            return FAIL, "setUp: " + traceback.format_exc()
        # Impose the gloabl-rank identity after setUp
        inst.rank = self._rank
        self._cur["key"] = tag
        self._cur["n"] = 0
        try:
            getattr(inst, test)()       # join_or_run -> rank != MAIN -> runs the real body
            return OK, ""
        except unittest.SkipTest as ex:
            return SKIP, str(ex)
        except SystemExit as ex:
            # MultiProcessTestCase skip decorator sys.exit(TEST_SKIPS[...].exit_code) instead of
            # raising, since normally each rank is its own process.
            code = getattr(ex, "code", None)
            entry = next((v for v in TEST_SKIPS.values() if v.exit_code == code), None)
            return (SKIP, entry.message) if entry else (FAIL, f"SystemExit(code)={code})")
        except Exception:
            return FAIL, traceback.format_exc()
        finally:
            try:
                inst.tearDown()
            except Exception:
                pass
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            os.environ.clear()
            os.environ.update(env_snapshot)

    def run_one_guarded(self, cls, test, tag):
        """
        This is run_one with a per-test wall-clock backstop. The PG-timeout clamp + CleanUpOnly watchdog should
        turn any collective desync into a fast catchable FAIL inside run_one.  This guard covers the residual case
        of a NON-NCCL hang (store.wait, CPU-side barrier, python deadlock) the watchdog can't see. Run the body in
        daemon thread and if it overruns args.test_timeout, abandon it. The thread stays stuck but it is a daemon
        so it won't block exit and reports a HANG. Because every rank entered this test together at the pre-barrier
        and shares the same wall-clock bound, peers give up within epsilon of each other and re-sync at the next
        control barrier (whose timeout is strictly larger).
        """
        if self._no_guard:
            return self.run_one(cls, test,tag)
        box = {}

        def target():
            box["r"] = self.run_one(cls, test, tag)

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(self._test_timeout)
        if th.is_alive():
            return FAIL, f"HANG: body did not return within {self._test_timeout}s wall-clock (rank {self._rank})"
        return box.get("r", (FAIL, "test thread produced no result"))

    def run_isolated_child(self, idx):
        """
        Run plan[idx] in a fresh throwaway subprocess and map its exit code to (status, detail). The
        child rebuilds the identical plan (sam module/class/test/exclude filters are forwarded) so
        plan[idx] is the same test on both ranks, and rendezvous-es over the same shared TCPStore.
        """
        child_argv = [sys.executable, __file__, "--run-single-index", str(idx),
                      "--module", self.module, "--pg-timeout", str(self._pg_timeout)]
        if self._test_file_pth:
            child_argv += ["--test-file", self._test_file_pth]
        if self._classes:
            child_argv += ["--classes", self._classes]
        if self._tests:
            child_argv += ["--tests", self._tests]
        if self._exclude:
            child_argv += ["--exclude", self._exclude]
        try:
            proc = subprocess.Popen(child_argv)
        except Exception as e:   # noqa: BLE001
            return FAIL, f"failed to spawn isolated child: {e}"
        # The child pays a cold start (import torch + CUDA/NCCL bootstrap) the in-process path doesn't,
        # so give it the wall-clock budget plus a startup margin befoer the parent SIGKILLs it.
        iso_timeout = self._test_timeout + 60
        try:
            rc = proc.wait(timeout=iso_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
            return FAIL, f"HANG: isolated child exceeded {iso_timeout}s wall-clock (rank {self._rank})"
        if rc == 0:
            return OK, ""
        if rc == 3:
            return SKIP, "isolated child reported skip"
        if rc < 0:
            return FAIL, f"CRASH: isolated child killed by signal {-rc} (rank {self._rank})"
        return FAIL, f"isolated child exit code {rc} (rank {self._rank})"
    
    @property
    def rank(self):
        return self._rank

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def world(self):
        return self._world

    @property
    def local_world(self):
        return self._local_world

    @property
    def master_addr(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def is_rank0(self):
        return self._is_rank0


def main():
    ap = argparse.ArgumentParser(description="Run upstream MultiProcessTestCase suites cross-node.")
    ap.add_argument("--module", default="test_c10d_nccl",
                    help="Upstream test module under test/distributed. Default: test_c10d_nccl")
    ap.add_argument("--test-file", default=None, help="Absolute path to the upstream test py file. Overrides --module")
    ap.add_argument("--classes", default=None, help="Comma separated list of class names (default: all wrappable in module).")
    ap.add_argument("--tests", default=None, help="Comma separated substrings that runs matching test methods.")
    ap.add_argument("--exclude", default=None, help="Comma separated substrings that skip matching test methods.")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Control barrier / TCPStore timeout seconds. The bound that a rank waits for its peer to finish a test.")
    ap.add_argument("--pg-timeout", type=int, default=90,
                    help="Clamp every test process-group's timeout. This will raise with TORCH_NCCL_BLOCKING_WAIT instead of hanging.")
    ap.add_argument("--test-timeout", type=int, default=150,
                    help="Per-test wall clock for tests that the PG clamp can't reach. Abandon test as a hang/fail and continue. "
                    "This fits between --pg-timeout and --timeout.")
    ap.add_argument("--no-guard", action="store_true", help="Disalbe the per-test wall-clock thread guard.")
    ap.add_argument("--no-poison", action="store_true", help="Run every test indepednently instead of skipping tests after a failure.")
    ap.add_argument("--isolate", action="store_true", help="Run every test in a fresh throwawy subprocess.")
    ap.add_argument("--no-isolate", action="store_true", help="Never isolate. Run all tests in-process for debugging.")
    ap.add_argument("--run-single-index", type=int, default=None, help=argparse.SUPPRESS) # INTERNAL: spawned per-teset child runs plan[N] then exists with a status code
    ap.add_argument("--list", action="store_true", help="List eligible classes/tests for current world and exits.")
    ap.add_argument("--pytorch-src", default=os.environ.get("PYTORCH_SRC"),
                           help="Path to a local pytorch checkout to pull test/distributed/<module>.py from. "
                                "(default: $PYTORCH_SRC; if unset, fetches from upstream GitHub)")
    ap.add_argument("--ref", default="main",
                           help="Upstream git ref (branch/tag/SHA) to fetch the test file from when no local "
                                "source is found. Default: main (latest).")
    args = ap.parse_args()

    # Keep the timeouts ordered. Ordered in ascending seconds
    # 1. collective clamp (--pg-timeout) 2. wall-clock (--test-timeout) 3. control barrier (--timeout)
    if args.test_timeout <= args.pg_timeout:
        args.test_timeout = args.pg_timeout + 60
    if args.timeout <= args.test_timeout:
        args.timeout = args.test_timeout + 120

    # Because of the nature of the tests by MultiProcessTestCase where, the device rank and global rank is tied
    # symmetrical gpu fabric is a no go. So GPUDirect needs to be turned off and the universal PIX needs to be utilized.
    os.environ.setdefault("NCCL_IB_HCA", "mlx5_0")
    os.environ.setdefault("NCCL_IB_GID_INDEX", "3")     # RoCE v2, IPv4-mapped GID
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "LOC")  # force host-staged RDMA

    # Hang proofing. Since each rank is its own process, stuck peers are just SIGKILLed at teardown under real multi-process
    # spawns. The tests are one long-lived global ranks running tests sequentially, so a peer stuck in a collective would hang
    # the whole run. Make the desync recover without taking the process down by using the async watchdog in CleanUpOnly mode.
    #       * TORCH_NCCL_BLOCKING_WAIT MUST be OFF. A blocking-wait timeout calls handleException(TearDown)
    #         UNCONDITIONALLY -> std::rethrow on the process -> the rank DIES and its peer hangs at the
    #         barrier -> JobSet SIGTERMs the run (observed twice). Blocking wait also suppresses the
    #         watchdog entirely ("NO watchdog thread is created"). So force it 0.
    #       * TORCH_NCCL_ASYNC_ERROR_HANDLING=2 (CleanUpOnly): on a collective timeout the watchdog ABORTS
    #         the stuck communicator (SHOULD_CLEAN_UP(2)=true) -- freeing the GPU so the next test is clean
    #         and making the test's pending op RAISE, which run_one catches as a clean FAIL -- but NEVER
    #         tears the process down (SHOULD_TEAR_DOWN(2)=false). Both ranks then re-sync at the next barrier.
    #         Why 2 and not 0 (NoHandling)? Tests that set TORCH_DISTRIBUTED_DEBUG=DETAIL (the *_debug_detail
    #         variants) flip dist_debug_level to Detail, and PGNCCL then UPGRADES NoHandling -> SkipCleanUp(3),
    #         which DOES tear down. CleanUpOnly is ineligible for that upgrade (guard is `== NoHandling`), so
    #         it is teardown-proof regardless of a test's debug level.
    #     We clamp every PG's timeout (below) to args.pg_timeout so the watchdog fires in seconds, not the
    #     10-min default, and the per-test wall-clock guard (further below) is the backstop for any hang the
    #     watchdog can't see (a non-NCCL / CPU-side deadlock). Hard-set (not setdefault): correctness depends
    #     on these EXACT values and a teardown is fatal to the whole run.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "2" # CleanUpOnly: abort comm, never tear process down

    # Neutralize spawn machinery to disable the forks and no setUp can accidentally spawn.
    MultiProcessTestCase._spawn_processes = lambda self: None
    MultiProcessTestCase._start_processes = lambda self, proc: None
    MultiProcessTestCase._join_processes = lambda self, fn: None

    cross_mpth = CrossMultiProcessTH(
        args.module,
        args.test_file,
        MultiProcessTestCase,
        args.no_guard,
        args.test_timeout,
        args.pg_timeout,
        args.classes,
        args.excludes,
        args.tests
        args.pytorch_src,
        args.ref
        )

    torch.cuda.set_device(cross_mpth.local_rank)

    path = cross_mpth.discover_test_file()
    only_classes = set(args.classes.split(",")) if args.classes else None
    substrings = args.tests.split(",") if args.tests else None
    excludes = args.excludes.split(",") if args.excludes else None

    mod = cross_mpth.import_test_module()
    classes = cross_mpth.collect_classes(only_classes)
    if only_classes and cross_mpth.is_rank0:
        unknown = only_classes - {n for n, _ in classes}
        blocked = only_classes & BLOCK_CLASSES
        if unknown - blocked:
            _log(f"Warning: unknown class name(s) ignored: {', '.join(sorted(unknown - blocked))}")
        if blocked:
            _log(f"NOTE: blocklisted (spawn/exit-code) class(es) skipped: {', '.join(sorted(blocked))}")
    if not classes:
        _log(f"no wrappable MultiProcessTestCase classes selected in {os.path.basename(path)} -- SKIP")
        os.exit(0)

    def world_size_of(cls, test):
        """Instantiate + run setUp, then read the world_size of this specific test."""
        inst = cls(test)
        try:
            inst.setUp()
        except BaseException:
            pass
        try:
            return int(inst.world_size), inst
        except Exception:
            return None, inst

    plan = []  # (cname, cls, test)
    for cname, cls in classes:
        for test in cross_mpth.collect_tests(cls, substrings, excludes):
            plan.append(cname, cls, test)

    if args.list:
        if cross_mpth.is_rank0:
            _log(f"# {os.path.basename(path)}: physical world={cross_mpth.world}"
                 f"({cross_mpth.local_world} proc/pod x {cross_mpth.world // max(cross_mpth.local_world, 1)} nodes, {cross_mpth.world} GPU/pod)")
            cur = None
            for cname, cls, test in plan:
                ws, _ = world_size_of(cls, test)
                mark = "run " if ws == cross_mpth.world else f"skip(w={ws})"
                if cname != cur:
                    _log(f"\n{cname}:")
                    cur = cname
                _log(f"    [{mark}] {test}")
        os._exit(0)

    # One extra persistent cross-node TCPStore on rank0, port = MASTER_PORT + 1 so it never clashes with
    # torchrun's own env:// store on MASTER_PORT. Both control (status/barrier) and every test's rendevous
    # ride this store via disjoint PrefixStores.
    store_is_master = cross_mpth.is_rank0 and args.run_single_index is None
    base_store = dist.TCPStore(
        cross_mpth.master_addr,
        cross_mpth.master_port + 1,
        world_size=cross_mpth.world,
        is_master = store_is_master,
        timeout = datetime.timedelta(seconds=args.timeout)
    )
    coord = StoreCoord(dist.PrefixStore("ctrl", base_store), cross_mpth.rank, cross_mpth.world, args.timeout)

    # Patch FileStore -> PrefixStore(<per-test key>, base_store). SPMD: every rank makes the same FileStore calls in the same order.
    _cur = {"key": "init", "n": 0}

    def _patched_filestore(file_name=None, world_size=-1, *args, **kwargs):
        _cur["n"] += 1
        return dist.PrefixStore(f'rdzv/{_cur["key"]}/{_cur["n"]}', base_store)

    for target in (dist, c10d):
        target.FileStore = _patched_filestore
    # the test module may hold it own reference (import torch.distributed as c10d /from ... import)
    if hasattr(mod, "c10d"):
        mod.c10d.FileStore = _patched_filestore
    if hasattr(mod, "FileStore"):
        mod.FileStore = _patched_filestore

    # Clamp every test PG's timeout so a desynced collective raises fast.
    _pg_to = datetime.timedelta(seconds=args.pg_timeout)

    def _clamp_timeout(kwargs):
        t = kwargs.get("timeout")
        if t is None or not isinstance(t, datetime.timedelta) or t > _pg_to:
            kwargs["timeout"] = _pg_to
        return kwargs

    def _wrap_pg_factory(orig):
        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
            return orig(*args, **_clamp_timeout(kwargs))
        return wrapped

    for target in (dist, c10d):
        for fname in ("init_process_group", "new_group"):
            fn = getattr(target, fname, None)
            if fn is not None and not hasattr(fn, "__wrapped__"):
                setattr(target, fname, _wrap_pg_factory(fn))
    # the test module may hold its own reference pulled in at import time
    for fname in ("init_process_group", "new_group"):
        if hasattr(mod, fname):
            setattr(mod, fname, _wrap_pg_factory(getattr(mod, fname)))
        if hasattr(mod, "c10d") and hasattr(mod.c10d, fname) and not hasattr(getattr(mod.c10d, fname), "__wrapped__"):
            setattr(mod.c10d, fname, _wrap_pg_factory(getattr(mod.c10d, fname)))

    def _best_effort_reset(limit=10):
        """
        Tear down a lingering default PG (e.g. left by an abandoned/hung body) without itself hanging.
        Destroy in a short-lived daemon thread and move on if it doesn't return.
        """
        if not dist.is_initialized():
            return

        def _destroy():
            try:
                dist.destroy_process_group()
            except Exception:
                pass

        th = threading.Thread(target=_destroy, daemon=True)
        th.start()
        th.join(limit)

    if cross_mpth.is_rank0 and args.run_single_index is None:
        _log(f"world_size={cross_mpth.world}  ({cross_mpth.local_world} proc/pod x {cross_mpth.world // max(cross_mpth.local_world, 1)} nodes)  "
             f"backend=nccl  module={os.path.basename(path)}  candidate_tests={len(plan)}")

    poisoned = set()
    results = []  # (cname, test, status, detail)

    # Isolated-child mode. Runs exactly one test in a throwaway process.
    if args.run_single_index is not None:
        idx = args.run_single_index
        if idx < 0 or idx >= len(plan):
            _log(f"isolated child: index {idx} out of range (plan has {len(plan)})")
            os._exit(1)
        _, cls_c, test_c = plan[idx]
        st, det = cross_mpth.run_one(cls_c, test_c, str(idx))
        if st == FAIL and det:
            _log(det.rstrip())
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit({OK: 0, SKIP: 3, FAIL: 1}[st])

    _self = os.path.abspath(__file__)

    def _isolated(cname):
        if args.no_isolate:
            return False
        if args.isolate:
            return True
        return cname in ISOLATED_CLASSES

    for idx, (cname, cls, test) in enumerate(plan):
        tag = str(idx)
        # eligibility (deterministic + identical on every rank -> lockstep): declared world must match.
        ws, _ = world_size_of(cls, test)
        if ws != cross_mpth.world:
            results.append((cname, test, SKIP, f"needs world {ws}. This launch has world {cross_mpth.world}"))
            continue
        # know in process incompatible individual tests
        if test in XNODE_INCOMPAT:
            reason = XNODE_INCOMPAT[test]
            results.append((cname, test, SKIP, reason))
            if cross_mpth.is_rank0:
                _log(f"[SKIP] {cname}.{test}  ({reason})")
            continue
        # source-level guard: a body that pokes the spawn machiner can't run in-process
        try:
            src = inspect.getsource(getattr(cls, test))
            if any(m in src for m in SKIP_SOURCE_MARKERS):
                results.append((cname, test, SKIP, "uses multi-process spawn/exit-code machinery"))
                continue
        except Exception:
            pass
        if cname in poisoned:
            results.append((cname, test, SKIP, "poisoned: earlier test in class failed."))
            continue

        _best_effort_reset()        # clear any default PG a prior hung body abandoned
        coord.barrier(f"pre/{tag}")
        if _isolated(cname):
            local_status, detail = cross_mpth.run_isolated_child(idx)
        else:
            local_status, detail = cross_mpth.run_one_guarded(cls, test, tag)
        status = coord.max_status(tag, local_status)   # worst outcome on any rank governs
        results.append((cname, test, status, detail))

        if status == FAIL:
            if not args.no_poison:
                poisoned.add(cname)
            if cross_mpth.is_rank0:
                _log(f"[FAIL] {cname}.{test}")
                if local_status == FAIL and detail:
                    _log(detail.rstrip())
        elif cross_mpth.is_rank0:
            _log(f"[{'PASS' if status == OK else 'SKIP'}] {cname}.{test}"
                 + (f"  ({detail})" if status == SKIP and detail else ""))

    npass = sum(1 for r in results if r[2] == OK)
    nskip = sum(1 for r in results if r[2] == SKIP)
    nfail = sum(1 for r in results if r[2] == FAIL)
    if cross_mpth.is_rank0:
        _log("\n" + "=" * 60)
        _log(f"CROSS-NODE UPSTREAM MultiProcessTestCase RUN  ({os.path.basename(path)}, world {cross_mpth.world})")
        _log(f"  {npass} passed, {nfail} failed, {nskip} skipped   (of {len(results)})")
        if nfail:
            _log("  FAILED:")
            for cname, test, s, _ in results:
                if s == FAIL:
                    _log(f"    {cname}.{test}")
        _log("ALL DONE" if nfail == 0 else "FAILURES ABOVE")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
