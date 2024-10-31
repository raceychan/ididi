import abc

import pytest

from ididi.graph import DependencyGraph

dag = DependencyGraph()


@dag.node
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


@pytest.mark.asyncio
async def test_close_graph_with_resources():
    await dag.aclose()


@pytest.mark.asyncio
async def test_resource():
    resource = dag.resolve(ArbitraryResource)
    assert not resource._closed
    await resource.close()
    assert resource._closed


@pytest.mark.debug
@pytest.mark.asyncio
async def test_graph_context_manager():

    async with dag:
        async with dag.resolve(ArbitraryResource) as resource:
            assert not resource._closed
    assert resource._closed
