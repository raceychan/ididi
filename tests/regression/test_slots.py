import pytest

from ididi._node import Dependencies, Dependency, DependentNode
from ididi.graph import Graph


async def test_graph_ds_slots():
    dg = Graph()
    with pytest.raises(AttributeError):
        dg.__dict__

    sc = dg.scope().__enter__()

    with pytest.raises(AttributeError):
        sc.__dict__

    asc = await dg.ascope().__aenter__()

    with pytest.raises(AttributeError):
        asc.__dict__


def test_node_ds_slots():
    class User: ...

    n = DependentNode.from_node(User)

    with pytest.raises(AttributeError):
        n.__dict__

    dep = Dependency(1, 2, 3, 4)
    with pytest.raises(AttributeError):
        dep.__dict__

    deps = Dependencies(dict(d=dep))

    with pytest.raises(AttributeError):
        deps.__dict__
