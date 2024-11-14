import pytest

from ididi.errors import ResourceOutsideScopeError
from ididi.graph import DependencyGraph

dag = DependencyGraph()


class ArbitraryResource:
    def __init__(self):
        self._closed = False

    async def close(self) -> None:
        self._closed = True

    async def open(self) -> None:
        self._closed = False

    async def __aenter__(self) -> "ArbitraryResource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    @property
    def is_closed(self) -> bool:
        return self._closed


@pytest.mark.asyncio
async def test_close_graph_with_resources():
    await dag.aclose()


@pytest.mark.asyncio
async def test_resource():
    async with dag:
        with pytest.raises(ResourceOutsideScopeError):
            await dag.aresolve(ArbitraryResource)


@pytest.mark.asyncio
async def test_graph_context_manager():

    async with dag.scope() as scope:
        resource = await scope.resolve(ArbitraryResource)
        async with resource:
            assert not resource.is_closed
    assert resource.is_closed
