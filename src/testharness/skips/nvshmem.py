# Transport/topology suits
# Certain nvshmem test are only valid with specific gpu placement. The tests are grouped accordingly.
#   internode --bucket 2gpu (1 GPU/pod x N nodes; every peer hop = NIC / IBRC-IBGDA)
#   mixed     --bucket 4gpu (2 GPU/pod x N nodes; P2P same-node + RDMA cross-node)
#   intranode intranode-nvlink.yml (2 GPU, signle node; NVLINK load/store)
#   subgroup  subgroup-nvlink.yml (4 GPU, single node)

BLOCK_CLASSES = {}

# nvshmem_ptr() load/store + NVLS multicast. Need this with NVSWITCH capabilities.
NVLINK_ONLY = ["get_remote", "multicast_ptr", "tile_reduce", "multi_root",
               "one_shot", "multimem"] # one_shot + multimem is in test_symmetric_memory
NEEDS_4_LOCAL_GPU = ["subgroup"]

# currently test_get is failing on 2 nodes, 1 GPU/pod.
XFAIL_WHEN_1GPU_PER_POD = {
    "test_get": (
        "symm_mem.get() into a non-symmetric (unregistered) local dst is unsupported cross-node "
        "(NVSHMEM getmem needs a registered local MR -> ibrc WC status 4, LOC_PROT_ERR). Passes "
        "where the peer is P2P-reachable. Upstream op bug in nvshmem_get_out; see the bug doc."
    ),
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

# Asymmetric NVLS tile reduce hangs. Need NVLS hardware. Also need NVSHMEM_MAX_TEAMS raised.
ASYMMETRIC_TILE_REDUCE_HANGS = ["test_tile_reduce", "root_ratio_2"]

# the logic is skip if these conditions are met.
SKIP_TESTS = {
    "get_remote": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires NVLINK_ONLY"
    },
    "multicast_ptr": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires NVLINK_ONLY"
    },
    "tile_reduce": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires NVLINK_ONLY"
    },
    "multi_root": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires NVLINK_ONLY"
    },
    "one_shot": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires NVLINK_ONLY"
    },
    "subgroup": {
        "ngpus": lambda x: x < 4,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires a local node with at least 4 GPUs. Cannot be multinodal."
    },
    "test_get": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x and y,
        "reason": "symm_mem.get() into a non-symmetric (unregistered) local dst is unsupported cross-node "
        "(NVSHMEM getmem needs a registered local MR -> ibrc WC status 4, LOC_PROT_ERR). "
        "Passes where the peer is P2P-reachable. Upstream op bug in nvshmem_get_out; see the bug doc."
    },
    "test_barrier": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    },
    "test_put_wait_signal": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    },
    "test_nvshmem_get": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Requires direct load/store access (NVLINK/P2P). NYI: Signaling network-connected peers."
    },
    "test_tile_reduce": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Needs NVLS hardware. Currently only testable through single node via NVLINK."
    },
    "root_ratio_2": {
        "ngpus": lambda x: x == 1,
        "nodes": lambda x: x > 1,
        "relation": lambda x, y: x or y,
        "reason": "Needs NVLS hardware. Currently only testable through single node via NVLINK."
    }
}


SUITES = {
    "internode": {
        "expect": "1gpu-per-node",   # local_world_size == 1 -> every peer is remote
        "topology": "1 GPU/pod x N nodes -- every peer hop crosses the NIC (IBRC/IBGDA)",
        "launch": "submit-job.py --bucket 2gpu",
        "only": None,                # all tests except the excludes below
        "exclude": [key for key, val in SKIP_TESTS.items() if val in ["NVLINK_ONLY", "NEEDS_4_LOCAL_GPU", "XFAIL_WHEN_1GPU_PER_POD"]]
    },
    "mixed": {
        "expect": "multi-gpu-multi-node",   # local_world>=2 and nnodes>=2
        "topology": "2 GPU/pod x N nodes -- P2P for same-node peers + RDMA cross-node",
        "launch": "submit-job.py --bucket 4gpu",
        "only": None,
        "exclude": NVLINK_ONLY + NEEDS_4_LOCAL_GPU,   # peer-ptr would DEADLOCK; NVLS would crash
        "exclude": [key for key, val in SKIP_TESTS.items() if val in ["NVLINK_ONLY", "NEEDS_4_LOCAL_GPU"]] # peer-ptr would DEADLOCK; NVLS would crash.
    },
    "intranode": {
        "expect": "single-node",     # nnodes == 1 -> peers are NVLink-reachable
        "topology": "2+ GPU, SINGLE node -- NVLink/NVSwitch load/store domain",
        "launch": "crossnode/intranode-nvlink.yml (torchrun --standalone --nnodes=1 --nproc_per_node=2)",
        "only": [key for key, val in SKIP_TESTS.items() if val == "NVLINK_ONLY"],         # run ONLY the NVLink-domain ops here
        "exclude": [key for key, val in SKIP_TESTS.items() if val in ["NVLINK_ONLY", "ASYMMETRIC_TILE_REDUCE_HANGS"]]
    },
    "subgroup": {
        "expect": "single-node-4gpu",   # nnodes == 1 AND local_world >= 4
        "topology": "4 GPU, SINGLE node -- DispatchCombineInSubgroups splits WORLD into "
                    ">=2 subgroups, so it needs >=4 local devices (skip_if_lt_x_gpu(4))",
        "launch": "crossnode/subgroup-nvlink.yml (torchrun --standalone --nnodes=1 --nproc_per_node=4)",
        "only": [key for key, val in SKIP_TESTS.items() if val == "NEEDS_4_LOCAL_GPU"],   # run ONLY the >=4-GPU subgroup test(s)
        "exclude": [],               # nothing else -- this is the one topology that CAN run them
    },
}
