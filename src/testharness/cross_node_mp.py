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

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_distributed import MultiProcessTestCase, TEST_SKIPS

from .config import HarnessConfig
from .testharness import CrossNodeTestHarnessBase, _log
from .skips.nccl import BLOCK_CLASSES, SKIP_SOURCE_MARKERS, ISOLATE_CLASSES as nccl_IC
from .skips.nvshmem import ISOLATE_CLASSES as nvshmem_IC


OK, SKIP, FAIL = 0, 1, 2

# Neutralize spawn machinery to disable the forks and no setUp can accidentally spawn.
MultiProcessTestCase._spawn_processes = lambda self: None
MultiProcessTestCase._start_processes = lambda self, proc: None
MultiProcessTestCase._join_processes = lambda self, fn: None


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
            ngpus,
            nodes,
            no_guard,
            test_timeout,
            pg_timeout,
            classes: list[str],
            tests: list[str] | None,
            ref: str = "main",
            repo: str = "pytorch",
            list_flag: bool = False,
            run_single_test: str | None = None,
            timeout: int = 600,
            no_poison: bool = False
        ):
        super().__init__(module, MultiProcessTestCase, classes, ngpus, nodes, ref, repo)
        # Patch FileStore -> PrefixStore(<per-test key>, base_store). SPMD: every rank makes the same FileStore calls in the same order.
        self._cur = {"key": "init", "n": 0}

        self._no_guard = no_guard
        self._test_timeout = test_timeout
        self._pg_timeout = pg_timeout
        self._timeout = timeout
        self._tests = tests         # the same thing as substrings
        self._list_flag = list_flag

        self._rank = int(os.environ["RANK"])
        self._local_rank = int(os.environ["LOCAL_RANK"])
        self._world = int(os.environ["WORLD_SIZE"])
        self._local_world = int(os.environ.get("LOCAL_WORLD_SIZE", self._world))
        self._master_addr = os.environ["MASTER_ADDR"]
        self._master_port = int(os.environ["MASTER_PORT"])
        self._is_rank0 = self._rank == 0
        self._run_single_test = run_single_test
        self._no_poison = no_poison

        self._poisoned = set()
        self._results = []  # (cname, test, status, detail)

        if self._only_classes and self._is_rank0:
            unknown = self._only_classes - {n for n, _ in self._classes}
            blocked = self._only_classes & BLOCK_CLASSES
            if unknown - blocked:
                _log(f"Warning: unknown class name(s) ignored: {', '.join(sorted(unknown - blocked))}")
            if blocked:
                _log(f"NOTE: blocklisted (spawn/exit-code) class(es) skipped: {', '.join(sorted(blocked))}")
        if not self._classes and self._is_rank0:
            _log(f"no wrappable MultiProcessTestCase classes selected in {os.path.basename(self._path)} -- SKIP")
            sys.exit(0)

        self._monkey_patch_mods()

    def _monkey_patch_mods(self):
        """
        One extra persistent cross-node TCPStore on rank0, port = MASTER_PORT + 1 so it never clashes with
        torchrun's own env:// store on MASTER_PORT. Both control (status/barrier) and every test's rendevous
        ride this store via disjoint PrefixStores.
        """
        # Clamp every test PG's timeout so a desynced collective raises fast.
        _pg_to = datetime.timedelta(seconds=self._pg_timeout)
        def _clamp_timeout(kwargs):
            t = kwargs.get("timeout")
            if t is None or not isinstance(t, datetime.timedelta) or t > datetime.timedelta(seconds=_pg_to):
                kwargs["timeout"] = _pg_to # will this scope cause issues???
            return kwargs

        def _wrap_pg_factory(orig):
            @functools.wraps(orig)
            def wrapped(*args, **kwargs):
                return orig(*args, **_clamp_timeout(kwargs))
            return wrapped

        store_is_master = self._is_rank0 and self._run_single_test is None
        self._base_store = dist.TCPStore(
            self._master_addr,
            self._master_port + 1,
            world_size=self._world,
            is_master=store_is_master,
            timeout=datetime.timedelta(seconds=self._timeout)
        )
        self._coord = StoreCoord(dist.PrefixStore("ctrl", self._base_store), self._rank, self._world, self._timeout)

        for target in (dist, c10d):
            target.FileStore = self._patched_filestore
        # the test module may hold its own reference (import torch.distributed as c10d /from ... import)
        if hasattr(self._mod, "c10d"):
            self._mod.c10d.FileStore = self._patched_filestore
        if hasattr(self._mod, "FileStore"):
            self._mod.FileStore = self._patched_filestored

        for target in (dist, c10d):
            for fname in ("init_process_group", "new_group"):
                fn = getattr(target, fname, None)
                if fn is not None and not hasattr(fn, "__wrapped__"):
                    setattr(target ,fname, _wrap_pg_factory(fn))
        # the test module may hold its own reference in at import time
        for fname in ("init_process_group", "new_group"):
            if hasattr(self._mod, fname):
                setattr(self._mod, fname, _wrap_pg_factory(getattr(self._mod, fname)))
            if hasattr(self._mod, "c10d") and hasattr(self._mod.c10d, fname) and not hasattr(getattr(self._mod.c10d, fname), "__wrapped__"):
                setattr(self._mod.c10d, fname, _wrap_pg_factory(getattr(self._mod.c10d, fname)))
        if store_is_master:
            _log(f"world_size={self._world}  ({self._local_world} proc/pod x {self._world // max(self._local_world, 1)} nodes)  "
                 f"backend=nccl  module={os.path.basename(self._path)}  candidate_tests={len(self._plan)}")
        return

    def _patched_filestore(self, file_name=None, world_size=1, *args, **kwargs):
        self._cur["n"] += 1
        return dist.PrefixStore(f"rdzv/{self._cur['key']}/{self._cur['n']}", self._base_store)

    def run(self):
        if self._run_single_test is not None:
            cname_c, _, test_c = self._run_single_test.partition(".")
            cls_c = next((c for n, c in self._classes if n == cname_c), None)
            if cls_c is None or not test_c or not hasattr(cls_c, test_c):
                _log(f"isolated child: could not resolve test {self._run_single_test!r}")
                return 1
            status, detail = self._run_one(cls_c, test_c, self._run_single_test)
            if status == FAIL and detail:
                _log(detail.rstrip())
            sys.stdout.flush()
            sys.stderr.flush()
            return {OK: 0, SKIP: 3, FAIL: 1}[status]

        for idx, (cname, cls, test) in enumerate(self._plan):
            tag = str(idx)
            # eligibility (deterministic + identical on every rank -> locksetp): declared world must match.
            ws, _ = world_size_of(cls, test)
            if ws != self._world:
                self._results.append((cname, test, SKIP, f"needs world {ws}. This launch has world {self._world}"))
                continue
            # source-level guard: a body that pokes the spawn machiner can't run in-process
            try:
                src = inspect.getsource(getattr(cls, test))
                if any(m in src for m in SKIP_SOURCE_MARKERS):
                    self._results.append((cname, test, SKIP, "uses multi-process spawn/exit-code machinery"))
                    continue
            except Exception:
                pass
            if cname in self._poisoned:
                self._results.append((cname, test, SKIP, "poisoned: earlier test in class failed."))
                continue

            _best_effort_reset()      # clear any default PG a prior hung body abandoned
            
            if self._is_rank0:
                _log(f"[RUN] {cname}.{test}")
            
            self._coord.barrier(f"pre/{tag}")
            if self._isolated(cname):
                local_status, detail = self._run_isolated_child(cname, test)
            else:
                local_status, detail = self._run_one_guarded(cls, test, tag)
            status = self._coord.max_status(tag, local_status)   # worst outcome on any rank governs
            self._results.append((cname, test, status, detail))

            if status == FAIL:
                if self._no_poison:
                    self._poisoned.add(cname)
                if self._is_rank0:
                    _log(f"[FAIL] {cname}.{test}")
                    if local_status == FAIL and detail:
                        _log(detail.rstrip())
            elif self._is_rank0:
                _log(f"[{'PASS' if status == OK else 'SKIP'}] {cname}.{test}"
                     + (f"  ({detail})" if status == SKIP and detail else ""))

        npass, nskip, nfail = 0, 0, 0
        for r in self._results:
            if r[2] == OK:
                npass += 1
            if r[2] == SKIP:
                nskip += 1
            if r[2] == FAIL:
                nfail += 1
        if self._is_rank0:
            _log("\n" + "=" * 60)
            _log(f"CROSS-NODE UPSTREAM MultiPorcessTestCase RUN  ({os.path.basename(self._path)}, world {self._world})")
            _log(f"  {npass} passed, {nfail} failed, {nskip} skipped   (of {len(self._results)})")
            if nfail:
                _log("  FAILED:")
                for cname, test, s, _ in self._results:
                    if s == FAIL:
                        _log(f"    {cname}.{test}")
            _log("ALL DONE" if nfail == 0 else "FAILURES ABOVE")
        sys.stdout.flush()
        sys.stderr.flush()

        # Synchronized teardown over the control store
        try:
            self._coord.barrier("final/teardown")
        except Exception:  # noqa: BLE001
            pass
        return 1 if nfail else 0

    def _run_one(self, cls, test, tag):
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

    def _run_one_guarded(self, cls, test, tag):
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
            return self._run_one(cls, test,tag)
        box = {}

        def target():
            box["r"] = self._run_one(cls, test, tag)

        th = threading.Thread(target=target, daemon=True)
        th.start()
        th.join(self._test_timeout)
        if th.is_alive():
            return FAIL, f"HANG: body did not return within {self._test_timeout}s wall-clock (rank {self._rank})"
        return box.get("r", (FAIL, "test thread produced no result"))

    def _run_isolated_child(self, cname, test):
        """
        Run one named test ("<Class>.<test>") in a fresh throwaway subprocess and map its exit code
        to (status, detail). The dchild re-emites the parent's own config with run_signle_test overriden,
        and rendezvous-es over the same shared TCPStore.
        """
        child_argv = [
            sys.executable, "-m", __spec__.name,
            "--module", self._mod_name,
            "--run-single-test", f"{cname}.{test}"
        ]
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

    def _isolated(self, cname):
        return cname in (nccl_IC | nvshmem_IC)
    
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
    ap.add_argument("--b64-cfg", type=str, help="Base 64 encoded json string of the harness config.")
    ap.add_argument("--module", type=str, default=None, help=argparse.SUPPRESS)          # isolated child only
    ap.add_argument("--run-single-test", type=str, default=None, help=argparse.SUPPRESS) # isolated child: "<Class>.<test>"
    args = ap.parse_args()

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

    if args.run_single_test:
        cfg = HarnessConfig(module=args.module, run_single_test=args.run_single_test)
    else:
        cfg = HarnessConfig.from_b64(args.b64_cfg)

    # Keep the timeouts ordered. Ordered in ascending seconds
    # 1. collective clamp (--pg-timeout) 2. wall-clock (--test-timeout) 3. control barrier (--timeout)
    if cfg.test_timeout <= cfg.pg_timeout:
        cfg.test_timeout = cfg.pg_timeout + 60
    if cfg.timeout <= cfg.test_timeout:
        cfg.timeout = cfg.test_timeout + 120
    
    cross_mpth = CrossMultiProcessTH(
        module=cfg.module, ngpus=cfg.ngpus, nodes=cfg.nodes,
        no_guard=cfg.no_guard, test_timeout=cfg.test_timeout, pg_timeout=cfg.pg_timeout,
        classes=cfg.classes, tests=cfg.tests, ref=cfg.ref, repo=cfg.repo,
        list_flag=cfg.list_flag, run_single_test=cfg.run_single_test, timeout=cfg.timeout,
        no_poison=cfg.no_poison
        )

    torch.cuda.set_device(cross_mpth.local_rank)
    code = cross_mpth.run()
    os._exit(code)

if __name__ == "__main__":
    main()
