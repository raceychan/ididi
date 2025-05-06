import pytest

from ididi import Graph, Ignore


def test_ignore_error():
    class Engine:
        def __init__(self, name: Ignore[str]):
            self.name = name


    dg = Graph()

    assert not dg.should_be_scoped(Engine)