import pytest

from ididi import DependencyGraph
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
