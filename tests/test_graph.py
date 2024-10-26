import pytest

from didi.graph import DependencyGraph
from didi.utils.pretty_utils import pretty_print

dag = DependencyGraph()


@dag.node
class Config:
    def __init__(self, env: str = "prod"):
        self.env = env


class Database:
    def __init__(self, config: Config):
        self.config = config


class Cache:
    def __init__(self, config: Config):
        self.config = config


class UserRepository:
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache


class AuthService:
    def __init__(self, db: Database):
        self.db = db


class UserService:
    def __init__(self, repo: UserRepository, auth: AuthService):
        self.repo = repo
        self.auth = auth


@dag.node
def auth_service_factory(database: Database) -> AuthService:
    return AuthService(db=database)


dag.node(UserService)


def test_dag_resolve():
    service = dag.resolve(UserService)
    assert isinstance(service.repo.db, Database)
    assert isinstance(service.repo.cache, Cache)
    assert isinstance(service.auth.db, Database)


def test_get_dependency_types():
    database_dependencies = dag.get_dependency_types(Database)
    assert len(database_dependencies) == 1
    assert Config in database_dependencies


def test_get_dependent_types():
    database_dependents = dag.get_dependent_types(Database)
    assert len(database_dependents) == 2
    assert UserRepository in database_dependents
    assert AuthService in database_dependents


def test_pretty_print():
    print("\n" + pretty_print(dag))


#     return dag, service
