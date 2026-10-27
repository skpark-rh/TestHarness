# Classes whose PASS/FAIL is defined by real multi-process spawn semantics. (Child exit codes,
# SIGABRT from watchdog, flight-recorder dumps). This is not runnable on one-process-per-rank. Skipped.
BLOCK_CLASS_REASON = "PASS/FAIL is defined by real multi-process spawn semantics. Not runnable on one-process-per-rank."
BLOCK_CLASSES = {
    "NCCLTraceTest": BLOCK_CLASS_REASON, 
    "NCCLTraceTestBase": BLOCK_CLASS_REASON, 
    "NCCLTraceTestDumpOnTimeoutBase": BLOCK_CLASS_REASON,
    "NCCLTraceTestDumpOnTimeout": BLOCK_CLASS_REASON, 
    "NCCLTraceTestDumpOnStuckRanks": BLOCK_CLASS_REASON,
    "NCCLTraceTestManyRanks": BLOCK_CLASS_REASON, 
    "NcclErrorHandlingTest": BLOCK_CLASS_REASON
}

# Test whose body reaches into the spawn machinery cant' run in-process. Skipped.
SKIP_SOURCE_MARKERS = (
    "self.processes", "_check_return_codes", "special_return_code_checks",
    "skip_return_code_checks", "self._spawn_processes"
)

# Individual tests that cannot run in the shared-process model even with hang-proofing because
# their desync hang holds the GIL. The CleanUpOnly watchdog aborts the stuck NCCL comm, but
# the test body is stuck in a GIL-holding call. PowerSGD spends the desync window inside its
# SVD/orthogonalization + a python future.wait() that never yields.
# Skip if ngpus and nodes is true.

GIL_HOLDING_DESYNC = {
    "ngpus": lambda x: True,
    "nodes": lambda x: True,
    "relation": lambda x, y: x or y,
    "match": "exact",
    "reason": "known in-process hang (GIL-holding desync) -- see upstream test comments"
}

INTRANODECOMM = {
    "ngpus": lambda x: x == 1,
    "nodes": lambda x: x > 1,
    "relation": lambda x, y: x or y,
    "match": "exact",
    "reason": "IntranodeComm is NVlink/same-node only."
}

NOTXNODE_RELAVENT = {
    "ngpus": lambda x: x == 1,
    "nodes": lambda x: x > 1,
    "relation": lambda x, y: x or y,
    "match": "exact",
    "reason": "Same-node NCCL config assertion; not cross-node meaningful."
}

SKIP_TESTS = {
    "test_powerSGD_ddp_comm_hook_nccl": GIL_HOLDING_DESYNC,
    "test_powerSGD_ddp_comm_hook_nccl_grad_is_view": GIL_HOLDING_DESYNC,
    "test_nccl_dist_backend_error": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "premise = both ranks share one CUDA device (ncclInvalidUsage); cross-node they are on different GPUs so the expected DistBackendError never raises."
    },
    "test_intra_node_comm_all_reduce_custom_group_name_False": INTRANODECOMM,
    "test_intra_node_comm_all_reduce_custom_group_name_True": INTRANODECOMM,
    "test_pass_nccl_options_config": NOTXNODE_RELAVENT,
    "test_pass_nccl_options_config_comm_name": NOTXNODE_RELAVENT
}
# Classes whose tests are run each in a fresh throwaway subprocess instead of in-process. ProcessGroupNCCLGroupTest
# drives low-level PG lifecycle back-to-back in one long-lived process. This accumulates NCCL communicator state that
# eventually double frees which leads to SIGSEGV.
ISOLATE_CLASSES = {"ProcessGroupNCCLGroupTest"}

# Upstream NCCL MultiProcContinuousTest suites.
# TODO: Put these under the skip dict with lambdas? And are these nccl tests??? They seem to be different.
MODULES = {
    "test_c10d_ops_nccl": dict(min_gpus_per_pod=2, min_world=2),
    "test_ce_colls": dict(min_gpus_per_pod=1, min_world=2),
    "test_nccl": dict(min_gpus_per_pod=1, min_world=2),
    "test_composability": dict(min_gpus_per_pod=2, min_world=4),
    "test_cupy_as_tensor": dict(min_gpus_per_pod=1, min_world=2)
}
