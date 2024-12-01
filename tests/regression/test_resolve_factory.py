import typing as ty

import pytest

from ididi import DependencyGraph


def test_graph_resolve_factory():
    class User:
        def __init__(self, age: int, address: str):
            self.age = age
            self.address = address

    def user_factory() -> ty.Generator[User, None, None]:
        u = User(1, "test")
        yield u

    dg = DependencyGraph()

    dg.static_resolve(user_factory)
    with dg.scope() as scope:
        u = scope.resolve(user_factory)

        assert u.age == 1
        assert u.address == "test"


@pytest.mark.asyncio
async def test_graph_resolve_complex_factory():
    dg = DependencyGraph()

    class User:
        def __init__(self, age: int, address: str):
            self.age = age
            self.address = address

    @dg.node(ignore=("address", float))
    async def user_factory(
        address: str, name: float = 3.4
    ) -> ty.AsyncGenerator[User, None]:
        u = User(1, address)
        yield u

    with pytest.raises(TypeError):
        async with dg.scope() as scope:
            u = await scope.resolve(user_factory)

    async with dg.scope() as scope:
        u = await scope.resolve(user_factory, address="a")
        assert u.address == "a"
