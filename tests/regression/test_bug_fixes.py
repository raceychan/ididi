import abc
import typing as ty
from typing import Annotated

import pytest

from ididi import Graph
from ididi.errors import (
    ABCNotImplementedError,
    ForwardReferenceNotFoundError,
    ProtocolFacotryNotProvidedError,
)

dg = Graph()


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

    with pytest.raises(ABCNotImplementedError):
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
    dag = Graph()

    class ServiceA:
        def __init__(self, b: "ServiceB"):
            self.b = b

    class ServiceB:
        def __init__(self, a: str = "a"):
            self.a = a

    with pytest.raises(ForwardReferenceNotFoundError):
        dag.resolve(ServiceA)


def test_factory_with_builtin_type_str():
    class Setttings:
        def __init__(self, name: str):
            self.name = name

    @dg.node
    def settings_factory(name: str = "test") -> Setttings:
        return Setttings(name)

    dg.resolve(Setttings)


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
        def __init__(self, a: ty.Union[int, str]):
            self.a = a

    with pytest.raises(TypeError):
        dg.resolve(Service)

    @dg.node
    def service_factory(abc: ty.Union[int, str] = 3) -> Service:
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
    assert str(node.dependencies)


def test_not_supported_type():
    @dg.node
    class AnyService:
        def __init__(self, a: str):
            self.a = a

    @dg.node
    def service_factory(ddd: ty.Any = 3) -> AnyService:
        return AnyService("test")

    dg.resolve(AnyService)


from ididi import AsyncResource, Graph, use


async def test_resource_across_scope():
    # given a resource Connection
    # a service that depnds on Connection

    class Connection:
        def __init__(self):
            self._opened = False
            self._closed = False

        async def open(self):
            self._opened = True

        async def close(self):
            self._closed = True

    async def conn_factory() -> AsyncResource[Connection]:
        conn = Connection()
        await conn.open()
        print("conn opened!")
        yield conn
        await conn.close()

    class Service:
        def __init__(self, conn: Annotated[Connection, use(conn_factory)]):
            self._conn = conn

    dg = Graph()

    async with dg.ascope() as scope1:
        s1 = await scope1.resolve(Service)

    async with dg.ascope() as scope2:
        s2 = await scope2.resolve(Service)

    assert s1 is not s2
    """
    This would cause error before 1.1.5 bug fix, s1 == s2
    because in graph.get_resolve_cache
    we have a fallback stategy if solution of dependent not found in scope
    then we check if it is in the solutions of graph

    since we only cache resolution to scope if dependent is a resource
    we might get the same resource across scopes.
    """
