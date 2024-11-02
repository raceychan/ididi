from pathlib import Path

import pytest

from ididi.graph import DependencyGraph
from ididi.visual import Visualizer


class Config:
    def __init__(self, value: str = "config"):
        self.value = value


class Engine:
    def __init__(self, config: Config):
        self.config = config


class Database:
    def __init__(self, engine: Engine):
        self.engine = engine


class AuthService:
    def __init__(self, database: Database):
        self.database = database


@pytest.fixture
def dag():
    return DependencyGraph()


def test_visualizer(tmp_path: Path):
    d = tmp_path / "test_visual"
    d.mkdir()
    p = d / "hello.png"

    dag = DependencyGraph()
    dag.node(AuthService)
    dag.static_resolve(AuthService)
    vis = Visualizer(dag)
    vis.view
    vis.make_graph().save(p)
