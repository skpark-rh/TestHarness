from abc import ABC, abstractmethod
from .discovery import DiscoverTests


def _log(*a):
    print(*a, flush=True)


class CrossNodeTestHarnessBase(ABC):
    def __init__(
        self,
        module: str,
        base,
        class_list: list[str],
        ngpus: int,
        nodes: int,
        ref: str = "main",
        repo: str = "pytorch"
    ) -> None:
        disc_test = DiscoverTests(module, ngpus, nodes, ref, repo)
        self._classes = disc_test.collect_classes(base)
        self._plan = disc_test.get_tests(base)
        self._mod = disc_test.mod
        self._mod_name = module

        self._only_classes = set(class_list) if class_list else None
        return

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _run_one(cls, test, tag):
        raise NotImplementedError
