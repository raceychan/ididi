from dataclasses import dataclass
from typing import Annotated

import pytest

from ididi import Graph, Ignore, NodeConfig, use


@dataclass
class User:
    name: str
    age: int


side_effect: list[int] = []


async def get_user() -> Ignore[User]:
    global side_effect
    side_effect.append(1)
    return User("test", 1)


class EP:
    def __init__(self, user: Annotated[User, use(get_user, reuse=False)]): ...


@pytest.mark.debug
async def test_resolve_func():
    dg = Graph()

    # # dg.analyze(EP)
    # dg.analyze(get_user, config=NodeConfig(reuse=False))
    # assert dg.nodes[get_user].config.reuse is False

    # dg.reset(clear_nodes=True)

    dg.node(get_user, reuse=False)
    assert dg.nodes[get_user].config.reuse is False
    dg.analyze(get_user)
    assert dg.nodes[get_user].config.reuse is False
