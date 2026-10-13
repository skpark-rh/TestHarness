from dataclasses import dataclass, field, fields, MISSING
import argparse, base64, json

BACKENDS = ("nccl_cont", "nccl_mpt", "nvshmem")

BACKEND_DEFAULTS = {
    "nccl_cont": dict(module="test_c10d_ops_nccl", timeout=300),
    "nccl_mpt":  dict(module="test_c10d_nccl",     timeout=600),
    "nvshmem":   dict(module="test_nvshmem",       timeout=300)
}


@dataclass
class HarnessConfig:
    # --- common selection knobs ---
    module: str | None = None
    ngpus: int = 1
    nodes: int = 2
    ref: str = "main"
    repo: str = "pytorch"
    classes: list[str] | None = None
    tests: list[str] | None = None
    timeout: int = 300
    no_poison: bool = False
    list_flag: bool = False

    # --- mpt only ---
    no_guard: bool = False
    test_timeout: int = 150
    pg_timeout: int = 90
    isolate: bool = False    # This seems to be a redundant flag.
    no_isolate: bool = False # why is there two???
    run_single_idx: int | None = None  # internal test

    # transport
    def to_b64(self) -> str:
        raw = json.dumps({f.name: getattr(self, f.name) for f in fields(self)}).encode()
        return base64.b64encode(raw).decode()

    @classmethod
    def from_b64(cls, blob: str) -> "HarnessConfig":
        return cls(**json.loads(base64.b64decode(blob).decode()))
    