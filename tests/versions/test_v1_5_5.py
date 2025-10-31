from dataclasses import dataclass
from typing import Annotated

import pytest

from ididi import Graph, Ignore, is_provided, use
from ididi.errors import ParamReusabilityConflictError, ConfigConflictError


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

class PP:
    def __init__(self, user: Annotated[User, use(get_user, reuse=True)]): ...
    

async def test_resolve_func():
    dg = Graph()

    dg.node(get_user)
    assert not is_provided(dg.nodes[get_user].reuse)
    dg.analyze(EP)
    assert dg.nodes[get_user].reuse is False

    with pytest.raises(ParamReusabilityConflictError):
        dg.analyze(PP)


async def test_resolve_dep_with_return():
    dg = Graph()

    class Manager: ...

    def get_manager() -> Annotated[Manager, use(reuse=True)]: ...

    
    dg.analyze(get_manager)
    assert dg.nodes[Manager].reuse

    with pytest.raises(ConfigConflictError):
        dg.node(Manager, reuse=False)


async def test_override_with_reusability():
    dg = Graph()

    @dg.node(reuse=False)
    class Manager: ...

    dg.analyze_nodes()
    assert not dg.nodes[Manager].reuse

    def get_manager() -> Annotated[Manager, use(reuse=True)]: ...
    dg.node(get_manager)
    dg.analyze_nodes()
    assert dg.nodes[Manager].reuse






