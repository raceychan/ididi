from typing import Annotated

from ididi import Graph, Ignore, use

type Q[T] = Annotated[T, "Q"]


def get_q(q: Q[str]) -> Ignore[int]: ...


async def create_user(q: Q[str], dep: Annotated[int, use(get_q)]) -> Ignore[int]:
    return q


async def test_ignore_alias():
    dg = Graph()

    node = dg.analyze(create_user, ignore=(Q,))

    assert await dg.resolve(create_user, q=5) == 5
