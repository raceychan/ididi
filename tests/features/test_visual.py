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


def test_visualizer():
    dag = DependencyGraph()
    dag.node(AuthService)
    vis = Visualizer(dag)
    vis.make_graph()
    vis.make_node(AuthService, {}, {})
