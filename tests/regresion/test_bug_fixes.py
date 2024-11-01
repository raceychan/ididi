import abc
import typing as ty

import pytest

from ididi import DependencyGraph
from ididi.errors import (
    ABCWithoutImplementationError,
    ForwardReferenceNotFoundError,
    ProtocolFacotryNotProvidedError,
    UnsolvableDependencyError,
)

dg = DependencyGraph()


def test_protocols():
    class Cache(ty.Protocol):
        def get(self, key: str) -> str: ...

    class MemoryCache:
        def get(self, key: str) -> str: ...

    @dg.node
    class Registry:
        def __init__(self, cache: Cache):
            self.cache = cache

    with pytest.raises(ProtocolFacotryNotProvidedError):
        dg.resolve(Registry)

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


# @pytest.mark.debug
def test_factory_with_builtin_type_str():
    class Setttings:
        def __init__(self, name: str):
            self.name = name

    @dg.node
    def settings_factory(name: str = "test") -> Setttings:
        return Setttings(name)

    dg.resolve(Setttings)


# @pytest.mark.debug
def test_factory_with_builtin_type():
    """
    BUG(fixed):
    DependencyParam:
        def from_param[
        I
    ](
        cls, dependent: type[I], param: inspect.Parameter, globalns: dict[str, ty.Any]
    ) -> "DependencyParam":
    ...
    return cls(
        name=param.name,
        param=param,
        dependency=dependency,
        is_builtin=is_builtin_type(param.annotation), <- BUG cause
    )
    for param like tuple[str, ...],  is_builtin is False
    """

    class Setttings:
        def __init__(self, name: str):
            self.name = name

    @dg.node
    def settings_factory(asdf: tuple[str, ...] = ("test",)) -> Setttings:
        return Setttings(asdf[0])

    settings = dg.resolve(Setttings)
    assert settings.name == "test"


def test_union_type():
    @dg.node
    class Service:
        def __init__(self, a: int | str):
            self.a = a

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(Service)

    @dg.node
    def service_factory(abc: int | str = 3) -> Service:
        return Service(3)

    dg.resolve(Service)


def test_complex_type():
    @dg.node
    class Service:
        def __init__(self, a: dict[str, int]):
            self.a = a

    @dg.node
    def service_factory(map: dict[str, int] = {"a": 3}) -> Service:
        return Service({"a": 3})

    dg.resolve(Service)

    node = dg.nodes[Service]
    assert node.factory is service_factory
    assert str(node.dependency_params)


def test_not_supported_type():
    @dg.node
    class AnyService:
        def __init__(self, a: str):
            self.a = a

    @dg.node
    def service_factory(ddd: ty.Any = 3) -> AnyService:
        return AnyService("test")

    # with pytest.raises(UnsolvableDependencyError):
    dg.resolve(AnyService)
