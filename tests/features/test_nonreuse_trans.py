import pytest

from ididi import Graph
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


def test_singleton_uses_transilient():
    dg = Graph()
    dg.node(reuse=False)(Database)
    with pytest.raises(ReusabilityConflictError):
        dg.analyze(AuthService)


def test_nonreuse_before_nonreuse():
    dg = Graph()
    dg.node(reuse=False)(Database)
    dg.node(reuse=False)(Repository)
    dg.node(reuse=False)(AuthService)
    dg.analyze(AuthService)


def test_nonreuse_before_reuse():
    dg = Graph()
    dg.node(reuse=False)(AuthService)
    dg.analyze(AuthService)
