from typing import Generator

from ididi import Graph

from ..test_data import UserService


def test_scope_resolve_fallback():
    graph = Graph()

    u = graph.resolve(UserService)

    with graph.scope() as scope:
        u2 = scope.resolve(UserService)

    assert UserService in graph._resolution_registry
    assert UserService not in scope._resolution_registry

    assert u is u2


def test_scope_resouce_fallback():
    graph = Graph()

    def user_factory(dg: Graph) -> Generator[UserService, None, None]:
        yield UserService(1, 2)

    u = graph.resolve(UserService)

    graph.node(user_factory)

    with graph.scope() as scope:
        u2 = scope.resolve(UserService)

    assert UserService not in graph._resolution_registry
    assert UserService in scope._resolution_registry

    assert u is not u2
