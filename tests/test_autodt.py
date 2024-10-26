from didi.node import DependencyNode


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


def test_node_types():
    node = DependencyNode.from_node(UserService)
    # print_dependency_tree(node)
    service = node.build()
    assert isinstance(service, UserService)
    assert isinstance(service.auth.db, Database)
