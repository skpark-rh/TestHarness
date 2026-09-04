import inspect
import os
import sys
import tempfile
import torch
import urllib.request
import urllib.error
from importlib.util import spec_from_file_location, module_from_spec
from torch.testing._internal.common_utils import instantiate_parametrized_tests

from abc import ABC, abstractmethod


def _log(*a):
    print(*a, flush=True)


class CrossNodeTestHarnessBase(ABC):
    UPSTREAM_RAW = "https://raw.githubusercontent.com/pytorch/pytorch/{ref}/test/distributed/{module}.py"

    def __init__(
        self,
        module: str,
        test_file: str | None,
        base: object,
        pytorch_src: str | None = None,
        ref: str = "main"
    ) -> None:
        self._mod_str = module
        self._test_file_pth = test_file
        self._base = base
        self._pytorch_src = pytorch_src
        self._ref = ref
        self._path = None
        self._mod = None

    @staticmethod
    @abstractmethod
    def run_one(cls, test, tag):
        raise NotImplementedError

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
        url = self.UPSTREAM_RAW.format(ref=self._ref, module=self._mod_str)
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
            if (inspect.isclass(obj) and issubclass(obj, self._base)
                    and obj is not self._base and obj.__module__ == self._mod.__name__):
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
