# import io
from pathlib import Path

# from unittest import mock


import pytest

pytest.importorskip("graphviz")

from ididi.graph import Graph
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
    return Graph()


def test_visualizer():
    dag = Graph()
    dag.node(AuthService)
    vis = Visualizer(dag)
    vis.make_graph()
    vis.make_node(AuthService, {}, {})


def test_complex_graph(tmp_path: Path):
    dg = Graph()
    vs = Visualizer(dg)

    class ConfigService:
        def __init__(self, env: str = "test"):
            self.env = env

    class DatabaseService:
        def __init__(self, config: ConfigService):
            self.config = config

    class CacheService:
        def __init__(self, config: ConfigService):
            self.config = config

    class BaseService:
        def __init__(self, db: DatabaseService):
            self.db = db

    class AuthService(BaseService):
        def __init__(self, db: DatabaseService, cache: CacheService):
            super().__init__(db)
            self.cache = cache

    class UserService:
        def __init__(self, auth: AuthService, db: DatabaseService):
            self.auth = auth
            self.db = db

    class NotificationService:
        def __init__(self, config: ConfigService):
            self.config = config

    class EmailService:
        def __init__(self, notification: NotificationService, user: UserService):
            self.notification = notification
            self.user = user

    dg.analyze(EmailService)
    vs.dot
    vs.view

    f = tmp_path / "vis"
    f.touch()  # Create the file
