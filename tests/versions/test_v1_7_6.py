from typing import Annotated

import pytest

from ididi import Graph, is_provided, use
from ididi.errors import ConfigConflictError, ParamReusabilityConflictError


class User: ...

def get_user() -> User:
    return User()


class EP:
    def __init__(self, user: Annotated[User, use(get_user, reuse=False)]): ...

class PP:
    def __init__(self, user: Annotated[User, use(get_user, reuse=True)]): ...
    

async def test_resolve_func():
    dg = Graph()

    dg.node(get_user)
    assert not is_provided(dg.nodes[User].reuse)

    dg.analyze(EP)
    assert dg.nodes[User].reuse is False

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



async def test_resolve_dep_with_missing():
    dg = Graph()

    @dg.node
    class Manager: ...

    assert not is_provided(dg.nodes[Manager].reuse)

    dg.node(Manager, reuse=False)

    with pytest.raises(ConfigConflictError):
        dg.node(Manager, reuse=True)

async def test_resolve_dep_with_reversed():
    dg = Graph()

    @dg.node
    class Manager: ...

    assert not is_provided(dg.nodes[Manager].reuse)

    dg.node(Manager, reuse=True)

    with pytest.raises(ConfigConflictError):
        dg.node(Manager, reuse=False)