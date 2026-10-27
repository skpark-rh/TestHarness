import inspect
import os
import urllib.request
import urllib.error
import sys
import tempfile
import torch
from importlib.util import spec_from_file_location, module_from_spec
from functools import wraps
from torch.testing._internal.common_distributed import MultiProcContinuousTest, MultiProcessTestCase

from .skips import nccl, nvshmem

def instantiate_parameterized_tests(generic_cls):
    """
    Instantiates tests that have been decorated with a parameterize_fn. This is generally performed by a 
    decorator subclass of _TestParametrizer.  The generic test will be replaced on the test class by
    parametrized tests with specialized names.  This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
    """
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # Remove the generic test from the test class.
        delattr(generic_cls, attr_name)

        # Add parametrized tests to the test class.
        def instantiate_test_helper(cls, name, test, param_kwargs):
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)

            if hasattr(generic_cls, name):
                raise AssertionError(f"Redefinition of test {name}")
            setattr(generic_cls, name, instantiated_test)

        for (test, test_suffix, param_kwargs, decorator_fn) in class_attr.parametrize_fn(
                class_attr, generic_cls=generic_cls, device_cls=None):
            full_name = f'{test.__name__}_{test_suffix}'

            # Apply decorators based on full param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    return generic_cls


class DiscoverTests:
    UPSTREAM_RAW = "https://raw.githubusercontent.com/{repo}/pytorch/{ref}/test/distributed/{module}.py"

    def __init__(
            self,
            mod_name: str,
            ngpus: int,
            nodes: int,
            ref: str = "main",
            repo: str = "pytorch"
            ):
        self._mod_name = mod_name
        self._ngpus = ngpus
        self._nodes = nodes
        self._ref = ref
        self._repo = repo
        self._path = ""
        self._mod = None
        if "nccl" in mod_name:
            self._skip_tests = nccl.SKIP_TESTS
        elif "nvshmem" in mod_name:
            self._skip_tests = nvshmem.SKIP_TESTS
        else:
            raise ValueError(f"Cannot find suite for module: {mod_name}!")
        
        self._discover_test_file()
        self._import_test_module()

    def _fetch_upstream(self) -> str:
        """
        Download the latest test/distributed/<module>.py from pytorch/pytorch to a temp file

        Defaults to the `main` branch. The newest test source and not the installed release.
        Override using ref.
        """
        url = self.UPSTREAM_RAW.format(repo=self._repo, ref=self._ref, module=self._mod_name)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            sys.exit(
                f"Could not fetch upstream test file `{self._mod_name}.py` from "
                f"pytorch/pytorch@{self._ref}: HTTP {e.code} ({url})\n"
            )
        except urllib.error.URLError as e:
            sys.exit(f"Could not reach GitHub to fetch `{self._mod_name}.py` ({url}): {e}")
        fd, tmp = tempfile.mkstemp(prefix=f"{self._mod_name}_", suffix=".py")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        print(f"fetched upstream {self._mod_name}.py from {self._repo}/pytorch@{self._ref}")
        return tmp

    def _local_test_file(self) -> str:
        """
        Path to test/distributed/<module>.py in the checkout this torch was built from.
        """
        src_root = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
        cand = os.path.join(src_root, "test", "distributed", f"{self._mod_name}.py")
        return cand if os.path.isfile(cand) else None
    
    def _discover_test_file(self):
        """
        Locate an upstream test file: local clone -> upstream fetch.
        """
        # fetch from upstream
        self._path = self._local_test_file() or self._fetch_upstream()
        return

    def _import_test_module(self):
        """Import an upstream test module by file path, with test/dirs on sys.path (as pytest would)."""
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
        return

    def collect_classes(self, base_cls):
        """
        Ordered list of base class subclasses defined in `mod`.
        """
        classes = []
        for name, obj in vars(self._mod).items():
            if (
                inspect.isclass(obj) and 
                issubclass(obj, base_cls) and
                obj is not base_cls and 
                obj.__module__ == self._mod.__name__ and
                nccl.BLOCK_CLASSES.get(name) is None and
                nvshmem.BLOCK_CLASSES.get(name) is None
            ):
                classes.append((name, obj))
        classes.sort(key=lambda kv: kv[0])
        return classes

    def collect_tests(self, class_obj) -> list:
        """Deterministic sorted test-method names for a class. Ensure the parameterized methods are expanded."""
        # Expand @parameterize template only for classes that weren't already decorated with
        # @instantiated_paramterized_tests at import. `instantiate_paramterized_tests` is not idempotent.
        def _has_raw_template(c):
            for n in dir(c):
                a = getattr(c, n , None)
                if hasattr(a, "parameterize_fn") and getattr(a, "__name__", None) == n:
                    return True
            return False
        if _has_raw_template(class_obj):
            instantiate_parameterized_tests(class_obj)
        names = sorted(n for n in dir(class_obj) if n.startswith("test"))
        return names

    def get_mpc_tests(self):
        """Grab all tests associated with the MultiProcContinuousTests."""
        return self.get_tests(MultiProcContinuousTest)

    def get_mp_tests(self):
        """Grab all tests associated with the MultiProcessTestCase"""
        return self.get_tests(MultiProcessTestCase)

    def get_tests(self, base_cls):
        """Helper function that grabs tests from distributed torch test modules."""
        def _skip_match(key, mode, t):
            if mode == "exact":
                return key == t
            return key in t
        
        tests = []
        for cname, cls in self.collect_classes(base_cls):
            for t in self.collect_tests(cls):
                skip = False
                for test_name, skip_dict in self._skip_tests.items():
                    if (
                        _skip_match(test_name, skip_dict.get("match", "substring"), t) and 
                        skip_dict["relation"](skip_dict["ngpus"](self._ngpus), skip_dict["nodes"](self._nodes))
                    ):
                        skip = True
                        reason = skip_dict["reason"]
                        print(f"[SKIP] {cname}.{t}  ({reason})")
                        break
                if skip:
                    continue
                tests.append((cname, cls, t))
        return tests

    @property
    def mod(self):
        return self._mod

    @property
    def path(self):
        return self._path
    