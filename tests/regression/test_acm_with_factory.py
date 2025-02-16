from contextlib import asynccontextmanager

import pytest

from ididi import AsyncResource, Graph
from ididi.errors import ResourceOutsideScopeError


class ACM:
    def __init__(self):
        self._closed = False

    def __aenter__(self):
        return self

    def __aexit__(self, *args):
        self._closed = True


async def test_resolve_acm():
    dg = Graph()

    with pytest.raises(ResourceOutsideScopeError):
        acm = await dg.aresolve(ACM)


async def test_resolve_with_factory():
    dg = Graph()

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
    assert dg.nodes[ACM].factory_type == "function"
    assert not acm._closed


async def test_user_defined_acm():
    @asynccontextmanager
    async def acm_factory() -> AsyncResource[ACM]:
        acm = ACM()
        yield acm

    dg = Graph()
    dg.analyze(acm_factory)
    acm_node = dg.nodes[ACM]
    assert acm_node.factory is acm_factory
    assert acm_node.factory_type == "aresource"
