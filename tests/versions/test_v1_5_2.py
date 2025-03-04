from typing import Annotated, TypeVar

from ididi import Graph, Ignore, Resolver, use

T = TypeVar("T")

Q = Annotated[T, "Q"]


def get_q(q: Q[str]) -> Ignore[int]: ...


async def create_user(q: Q[str], dep: Annotated[int, use(get_q)]) -> Ignore[int]:
    return q


async def test_ignore_alias():
    dg = Graph()

    node = dg.analyze(create_user, ignore=(Q,))

    assert await dg.resolve(create_user, q=5) == 5


import pytest


@pytest.mark.debug
async def test_resolve_resolver():
    dg = Graph()

    current_dg = dg.resolve(Resolver)

    async with dg.ascope() as asc:
        asc_rv = await asc.resolve(Resolver)
