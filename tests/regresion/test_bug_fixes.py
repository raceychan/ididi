import abc
import typing as ty

import pytest

from ididi import DependencyGraph
from ididi.errors import (
    ABCWithoutImplementationError,
    ForwardReferenceNotFoundError,
    ProtocolFacotryNotProvidedError,
)

dg = DependencyGraph()


class Cache(ty.Protocol):
    def get(self, key: str) -> str: ...


class MemoryCache:
    def get(self, key: str) -> str: ...


@dg.node
class Registry:
    def __init__(self, cache: Cache):
        self.cache = cache


def test_protocols():
    with pytest.raises(ProtocolFacotryNotProvidedError):
        dg.resolve(Registry)


def test_factory_protocols():
    @dg.node
    def cache_factory() -> Cache:
        return MemoryCache()

    dg.resolve(Registry)


def test_abc():
    class AbstractEngine(abc.ABC):
        @abc.abstractmethod
        def run(self) -> None: ...

    @dg.node
    class Database:
        def __init__(self, engine: AbstractEngine):
            self.engine = engine

    with pytest.raises(ABCWithoutImplementationError):
        dg.resolve(Database)


def test_abc_with_implementation():

    class AbstractEngine(abc.ABC):
        @abc.abstractmethod
        def run(self):
            pass

    @dg.node
    class Engine(AbstractEngine):
        def run(self): ...

    dg.resolve(AbstractEngine)


@pytest.mark.skip("TODO: implement abstract dependency")
def test_abc_dependency_with_implementation():
    class AbstractEngine(abc.ABC):
        @abc.abstractmethod
        def run(self):
            pass

    @dg.node
    class Engine(AbstractEngine):
        def run(self): ...

    @dg.node
    class Database:
        def __init__(self, engine: AbstractEngine):
            self.engine = engine

    dg.resolve(Database)


def test_forward_ref_in_local_scope():
    """
    Test that defining forward reference in local scope raises ForwardReferenceNotFoundError
    """
    dag = DependencyGraph()

    @dag.node
    class ServiceA:
        def __init__(self, b: "ServiceB"):
            self.b = b

    @dag.node
    class ServiceB:
        def __init__(self, a: str = "a"):
            self.a = a

    with pytest.raises(ForwardReferenceNotFoundError):
        dag.resolve(ServiceA)
