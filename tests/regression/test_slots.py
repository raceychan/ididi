import pytest

from ididi._node import Dependencies, Dependency, DependentNode
from ididi.graph import AsyncScope, Graph, ScopeManager, SyncScope


def test_graph_ds_slots():
    dg = Graph()
    sm = ScopeManager(1, 2, 3, 4)
    asc = AsyncScope(
        graph_resolutions=1,
        graph_singletons=2,
        name=2,
        pre=3,
        nodes=4,
        resolved_nodes=5,
        type_registry=6,
        ignore=7,
    )

    sc = SyncScope(
        graph_resolutions=1,
        graph_singletons=9,
        name=2,
        pre=3,
        nodes=4,
        resolved_nodes=5,
        type_registry=6,
        ignore=7,
    )

    with pytest.raises(AttributeError):
        dg.__dict__

    with pytest.raises(AttributeError):
        sm.__dict__

    with pytest.raises(AttributeError):
        asc.__dict__

    with pytest.raises(AttributeError):
        sc.__dict__


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
