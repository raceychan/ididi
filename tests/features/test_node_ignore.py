import pytest

from ididi import Graph, Ignore


class IgnoreNode:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


def test_direct_resolve_fail():
    dg = Graph()
    dg.node(IgnoreNode)
    dg.analyze(IgnoreNode)


def test_resolve_fail_with_partial_ignore():
    dg = Graph()

    dg.node(ignore=("name",))(IgnoreNode)
    dg.analyze(IgnoreNode)


def test_resolve_with_ignore():
    dg = Graph()
    dg.node(ignore=("name", int))(IgnoreNode)
    dg.analyze(IgnoreNode)


def test_resolve_without_ignore():
    dg = Graph()
    dg.node(ignore=("name", int))(IgnoreNode)

    with pytest.raises(TypeError):
        dg.resolve(IgnoreNode)

    n = dg.resolve(IgnoreNode, name="test", age=3)
    assert n.name == "test"
    assert n.age == 3


def test_resolve_with_positinoal_ignore():
    dg = Graph()
    dg.node(ignore=(0, "age"))(IgnoreNode)
    assert len(dg.nodes[IgnoreNode].dependencies) == 2

    i = dg.resolve(IgnoreNode, name="r", age=5)
    assert i.name == "r" and i.age == 5


def test_ignore_dependences():
    dg = Graph()

    class User:
        def __init__(self, name: Ignore[str]): ...

    dg.node(User)

    assert len(dg.nodes[User].dependencies) == 1
