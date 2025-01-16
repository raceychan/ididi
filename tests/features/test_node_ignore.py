import pytest

from ididi import DependencyGraph
from ididi.errors import UnsolvableDependencyError


class IgnoreNode:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


def test_direct_resolve_fail():
    dg = DependencyGraph()
    dg.node(IgnoreNode)
    with pytest.raises(UnsolvableDependencyError):
        dg.analyze(IgnoreNode)


def test_resolve_fail_with_partial_ignore():
    dg = DependencyGraph()

    dg.node(ignore=("name",))(IgnoreNode)
    with pytest.raises(UnsolvableDependencyError):
        dg.analyze(IgnoreNode)


def test_resolve_with_ignore():
    dg = DependencyGraph()
    dg.node(ignore=("name", int))(IgnoreNode)
    dg.analyze(IgnoreNode)


def test_resolve_without_ignore():
    dg = DependencyGraph()
    dg.node(ignore=("name", int))(IgnoreNode)
    with pytest.raises(TypeError):
        dg.resolve(IgnoreNode)

    n = dg.resolve(IgnoreNode, name="test", age=3)
    assert n.name == "test"
    assert n.age == 3


def test_resolve_with_positinoal_ignore():
    dg = DependencyGraph()
    dg.node(ignore=(0, "age"))(IgnoreNode)
    i = dg.resolve(IgnoreNode, name="r", age=5)
    assert i.name == "r" and i.age == 5
