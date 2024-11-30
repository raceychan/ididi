import pytest

from ididi import DependencyGraph
from ididi.errors import UnsolvableDependencyError


class IgnoreNode:
    def __init__(self, name: str, age: int): ...


def test_direct_resolve_fail():
    dg = DependencyGraph()
    dg.node(IgnoreNode)
    with pytest.raises(UnsolvableDependencyError):
        dg.static_resolve(IgnoreNode)


def test_resolve_fail_with_partial_ignore():
    dg = DependencyGraph()

    dg.node(ignore=("name",))(IgnoreNode)
    with pytest.raises(UnsolvableDependencyError):
        dg.static_resolve(IgnoreNode)


def test_resolve_with_ignore():
    dg = DependencyGraph()
    dg.node(ignore=("name", int))(IgnoreNode)
    dg.static_resolve(IgnoreNode)
