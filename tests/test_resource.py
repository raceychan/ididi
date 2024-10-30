import pytest

from ididi.graph import DependencyGraph

dag = DependencyGraph()

@dag.node
class Resource:
    def __init__(self):
        self._closed = False

    async def close(self) -> None:
        self._closed = True


@pytest.mark.asyncio
async def test_close_graph_with_resources():
    await dag.aclose()


@pytest.mark.asyncio
async def test_resource():
    resource = dag.resolve(Resource)
    assert not resource._closed
    await resource.close()
    assert resource._closed
