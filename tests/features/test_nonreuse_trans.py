import pytest

from ididi import DependencyGraph
from ididi.errors import ReusabilityConflictError


class Config:
    def __init__(self, file: str = "file"):
        self.file = file


class Database:
    def __init__(self, config: Config):
        self.config = config


class Repository:
    def __init__(self, db: Database):
        self.db = db


class AuthService:
    def __init__(self, repo: Repository):
        self.repo = repo


def test_reuse_before_nonreuse_error():
    dg = DependencyGraph()
    dg.node(reuse=False)(Database)
    with pytest.raises(ReusabilityConflictError):
        dg.static_resolve(AuthService)


def test_nonreuse_before_nonreuse():
    dg = DependencyGraph()
    dg.node(reuse=False)(Database)
    dg.node(reuse=False)(Repository)
    dg.node(reuse=False)(AuthService)
    dg.static_resolve(AuthService)


def test_nonreuse_before_reuse():
    dg = DependencyGraph()
    dg.node(reuse=False)(AuthService)
    dg.static_resolve(AuthService)
