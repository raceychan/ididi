from typing import Any

import pytest

from ididi import DependencyGraph, use
from ididi.errors import UnsolvableDependencyError


def test_graph_partial_resolve():
    dg = DependencyGraph(partial_resolve=True)

    class Data:
        def __init__(self, name: str):
            self.name = name

    dg.node(Data)

    dg.static_resolve(Data)

    with pytest.raises(TypeError):
        data = dg.resolve(Data)

    data = dg.resolve(Data, name="data")
    assert data.name == "data"


async def test_graph_partial_aresolve():
    dg = DependencyGraph(partial_resolve=True)

    class Data:
        def __init__(self, name: str):
            self.name = name

    dg.node(Data)

    dg.static_resolve(Data)

    with pytest.raises(TypeError):
        data = await dg.aresolve(Data)

    data = await dg.aresolve(Data, name="data")
    assert data.name == "data"


def test_graph_partial_resolve_with_dep():

    class Database: ...

    DATABASE = Database()

    def db_factory() -> Database:
        return DATABASE

    class Data:
        def __init__(self, name: str, db: Database = use(db_factory)):
            self.name = name
            self.db = db

    dg = DependencyGraph()

    with pytest.raises(UnsolvableDependencyError):
        dg.static_resolve(Data)

    partial_dg = DependencyGraph(partial_resolve=True)
    partial_dg.static_resolve(Data)

    with pytest.raises(TypeError):
        data = partial_dg.resolve(Data)

    data = partial_dg.resolve(Data, name="data")
    assert data.name == "data"
    assert data.db is DATABASE


def test_feat():
    dg = DependencyGraph()

    class T:
        def __init__(self, a: Any) -> None:
            self.a = a

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(T)

    dg = DependencyGraph(partial_resolve=True)

    t = dg.resolve(T, a=1)
