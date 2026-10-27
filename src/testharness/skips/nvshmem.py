# Transport/topology suits
# Certain nvshmem test are only valid with specific gpu placement. The tests are grouped accordingly.
#   internode --bucket 2gpu (1 GPU/pod x N nodes; every peer hop = NIC / IBRC-IBGDA)
#   mixed     --bucket 4gpu (2 GPU/pod x N nodes; P2P same-node + RDMA cross-node)
#   intranode intranode-nvlink.yml (2 GPU, signle node; NVLINK load/store)
#   subgroup  subgroup-nvlink.yml (4 GPU, single node)

BLOCK_CLASSES = {}

NVLINK_REQ = {
    "ngpus": lambda x: x == 1,
    "nodes": lambda x: x > 1,
    "relation": lambda x, y : x or y,
    "match": "substring",
    "reason": "Requires NVLINK_ONLY"
}

NVLINK_P2P_NYI = {
    "ngpus": lambda x: x == 1,
    "nodes": lambda x: x > 1,
    "relation": lambda x, y: x or y,
    "match": "exact",
    "reason": "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
}

# the logic is skip if these conditions are met.
SKIP_TESTS = {
    "get_remote": NVLINK_REQ,
    "multicast_ptr": NVLINK_REQ,
    "tile_reduce": NVLINK_REQ,
    "multi_root": NVLINK_REQ,
    "one_shot": NVLINK_REQ,
    "subgroup": {
        "ngpus": lambda x: x < 4,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "substring",
        "reason": "Requires a local node with at least 4 GPUs. Cannot be multinodal."
    },
    "test_get": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x and y,
        "match": "exact",
        "reason": "symm_mem.get() into a non-symmetric (unregistered) local dst is unsupported cross-node "
        "(NVSHMEM getmem needs a registered local MR -> ibrc WC status 4, LOC_PROT_ERR). "
        "Passes where the peer is P2P-reachable. Upstream op bug in nvshmem_get_out; see the bug doc."
    },
    "test_barrier": NVLINK_P2P_NYI,
    "test_put_wait_signal": NVLINK_P2P_NYI,
    "test_nvshmem_get": NVLINK_P2P_NYI,
    "test_tile_reduce": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "substring",
        "reason": "Needs NVLS hardware. Currently only testable through single node via NVLINK."
    },
    "root_ratio_2": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "substring",
        "reason": "Needs NVLS hardware. Currently only testable through single node via NVLINK."
    },
    "test_nvshmem_put": NVLINK_P2P_NYI,
    "test_nvshmem_all_to_all": NVLINK_P2P_NYI,
    # NOTE: these two match by op, not by name -- the NVLS/signaling keyword is in the test BODY,
    # not the method name, so the substring skip above would miss them and they run cross-node.
    "test_cuda_graph_collective": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "Captures torch.ops.symm_mem.one_shot_all_reduce (NVLS/multicast, NVLink-domain only) "
        "into a CUDA graph. Cross-node the multicast ptr is null -> device-side illegal memory access, "
        "surfaced at the NVSHMEM host barrier (barrier.cu) -> SIGSEGV. Unrecoverable native CUDA-context "
        "abort; cannot be caught/xfailed in-process, so it must be excluded."
    },
    "test_mempool_tensor_w_collective": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "dist.all_reduce on a symm-mem/NVLS-registered mempool tensor dispatches to the "
        "NVLS one-shot all-reduce (nvshmemi_tile_allreduce_nvls). Cross-node the multicast ptr is "
        "null -> device-side assert (reduce.cuh: src_v != nullptr) -> async SIGSEGV. NVLS/multicast "
        "is NVLink-domain only; uncatchable native abort, so it must be excluded."
    },
    "test_mempool_compute_ops": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "Calls torch.ops.symm_mem.nvshmem_broadcast on a symm-mem tensor; on this build the "
        "NVSHMEM broadcast lowers to the NVLS multicast kernel (nvshmemi_tile_allreduce_nvls). "
        "Cross-node the multicast ptr is null -> device-side assert -> SIGSEGV. NVLink-domain only."
    },
    "test_mempool_tensor_factory": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "Calls torch.ops.symm_mem.nvshmem_broadcast on a symm-mem tensor; on this build the "
        "NVSHMEM broadcast lowers to the NVLS multicast kernel (nvshmemi_tile_allreduce_nvls). "
        "Cross-node the multicast ptr is null -> device-side assert -> SIGSEGV. NVLink-domain only."
    },
    "test_team_pool_released_on_process_group_destroy": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "match": "exact",
        "reason": "Calls torch.ops.symm_mem.tile_reduce (NVLS tile reduce). Cross-node the multicast "
        "ptr is null -> device-side assert (reduce.cuh: src_v != nullptr) -> SIGSEGV. NVLink-domain only."
    },
    "test_barrier_timeout": NVLINK_P2P_NYI,
    "test_wait_signal_timeout": NVLINK_P2P_NYI,
    "test_put_signal_timeout": NVLINK_P2P_NYI,
    "test_channel_out_of_bounds": NVLINK_P2P_NYI
}

ISOLATE_CLASSES = {"NVSHMEMSignalNegativeTest"} # this determines the isolation execution method for multiprocesstest.
