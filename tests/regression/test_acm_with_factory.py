from contextlib import asynccontextmanager

import pytest

from ididi import AsyncResource, DependencyGraph
from ididi.errors import ResourceOutsideScopeError


class ACM:
    def __init__(self):
        self._closed = False

    def __aenter__(self):
        return self

    def __aexit__(self, *args):
        self._closed = True


async def test_resolve_acm():
    dg = DependencyGraph()

    with pytest.raises(ResourceOutsideScopeError):
        acm = await dg.aresolve(ACM)


async def test_resolve_with_factory():
    dg = DependencyGraph()

    async def acm_factory() -> ACM:
        return ACM()

    dg.node(acm_factory)

    acm = await dg.aresolve(ACM)

    assert not acm._closed

    def sync_factory() -> ACM:
        return ACM()

    dg.reset(clear_nodes=True)
    dg.node(sync_factory)

    acm = dg.resolve(ACM)
    assert not acm._closed


async def test_user_defined_acm():

    @asynccontextmanager
    async def acm_factory() -> AsyncResource[ACM]:
        acm = ACM()
        yield acm

    dg = DependencyGraph()
    dg.static_resolve(acm_factory)
    assert dg.nodes[ACM].factory is acm_factory
    assert dg.nodes[ACM].factory_type == "resource"
