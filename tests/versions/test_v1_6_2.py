from typing import Union

import pytest

from ididi import Graph, Ignore


class MyConfig: ...

def myfact(a: Ignore[Union[int, str, None]] = None) -> MyConfig:
    return MyConfig()


def test_resolve():
    graph = Graph()
    graph.node(myfact)
    node = graph.analyze(myfact)
    res = graph.resolve(MyConfig)
    assert isinstance(res, MyConfig)
