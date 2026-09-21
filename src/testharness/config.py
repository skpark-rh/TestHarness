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
    backend: str                    # nccl_cont | nccl_mpt | nvshmem

    # --- common selection knobs ---
    module: str | None = None
    test_file: str | None = None
    pytorch_src: str = ""
    ref: str = "main"
    classes: list[str] | None = None
    tests: list[str] | None = None
    excludes: list[str] | None = None
    timeout: int = 300
    no_poison: bool = False
    list_flag: bool = False

    # --- mpt only ---
    no_guard: bool = False
    test_timeout: int = 150
    pg_timeout: int = 90
    isolate: bool = False    # This seems to be a redundant flag.
    no_isolate: bool = False # why is there two???
    run_single_index: int | None = None  # internal test

    # --- nvshmem ---
    suite: str | None = None

    # transport
    def to_b64(self) -> str:
        raw = json.dumps({f.name: getattr(self, f.name) for f in fields(self)}).encode()
        return raw

    @classmethod
    def from_b64(cls, blob: str) -> "HarnessConfig":
        return cls(**json.loads(base64.b64decode(blob).decode()))
    