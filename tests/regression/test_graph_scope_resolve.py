from typing import Generator

from ididi import Graph

from ..test_data import UserService


def test_scope_resolve_fallback():
    graph = Graph()

    u = graph.resolve(UserService)

    with graph.scope() as scope:
        u2 = scope.resolve(UserService)

    assert UserService in graph._resolved_instances
    assert UserService not in scope._resolved_instances

    assert u is u2


def test_scope_resouce_fallback():
    graph = Graph()

    def user_factory(dg: Graph) -> Generator[UserService, None, None]:
        yield UserService(1, 2)

    u = graph.resolve(UserService)

    graph.node(user_factory)

    with graph.scope() as scope:
        u2 = scope.resolve(UserService)

    assert UserService not in graph._resolved_instances
    assert UserService in scope._resolved_instances

    assert u is not u2


class Service:
    def __init__(self, name: str = "s", age: int = 1): ...


async def test_shared_registered_singleton():
    dg = Graph()

    singleton = Service("service", 1)
    dg.register_singleton(singleton)

    scope_mng = dg.scope()

    with scope_mng as scope:
        service1 = scope.resolve(Service)

    async with scope_mng as ascope:
        service2 = await ascope.resolve(Service)

    assert service1 is singleton
    assert service2 is singleton


async def test_no_reversed_share():
    dg = Graph()

    service = Service("1", 2)
    with dg.scope() as scope:
        scope.register_singleton(service)
        assert scope.resolve(Service) is service

    assert not dg.is_registered_singleton(Service)
    s2 = dg.resolve(Service)
    assert s2 is not service
