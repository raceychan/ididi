import pytest

from ididi.errors import ForwardReferenceNotFoundError
from ididi.graph import Graph

# Create test classes with various dependency patterns


class ConfigService:
    def __init__(self, env: str = "test"):
        self.env = env


class CacheService:
    def __init__(self, config: ConfigService):
        self.config = config


class BaseService:
    def __init__(self, db: "DatabaseService"):
        self.db = db


class AuthService(BaseService):
    def __init__(self, db: "DatabaseService", cache: "CacheService"):
        super().__init__(db)
        self.cache = cache


class UserService:
    def __init__(self, auth: AuthService, db: "DatabaseService"):
        self.auth = auth
        self.db = db


class DatabaseService:
    def __init__(self, config: ConfigService):
        self.config = config


class NotificationService:
    def __init__(self, config: ConfigService):
        self.config = config


class EmailService:
    def __init__(self, notification: NotificationService, user: UserService):
        self.notification = notification
        self.user = user


@pytest.fixture(scope="function")
def dg() -> Graph:
    return Graph()


def test_analyze(dg: Graph):
    _ = dg.analyze(EmailService)
    assert len(dg.nodes) == 8
    assert len(dg.resolved_nodes) == 7


def test_analyze_equal_resolve(dg: Graph):
    _ = dg.analyze(EmailService)
    assert len(dg.nodes) == 8
    assert len(dg.resolved_nodes) == 7

    dg.reset()

    dg.resolve(EmailService)

    assert len(dg.nodes) == 8
    assert len(dg.resolved_nodes) == 7


class ForwardService:
    def __init__(self, client: "ForwardClient"):
        self.client = client


class ForwardClient:
    def __init__(self, config: "ForwardConfig"):
        self.config = config


class ForwardConfig:
    pass


def test_forward_dependency(dg: Graph):
    _ = dg.analyze(ForwardService)
    assert len(dg.nodes) == 4
    assert len(dg.resolved_nodes) == 3

    dg.resolve(ForwardService)
    assert len(dg.nodes) == 4
    assert len(dg.resolved_nodes) == 3


@pytest.mark.asyncio
async def test_async_enter(dg: Graph):
    @dg.node
    class AsyncService:
        pass

    dg.resolve(AsyncService)


def test_forward_ref_in_local_scope():
    dag = Graph()

    class ServiceA:
        def __init__(self, b: "ServiceB"):
            self.b = b

    class ServiceB:
        def __init__(self, a: str = "a"):
            self.a = a

    with pytest.raises(ForwardReferenceNotFoundError):
        dag.analyze(ServiceA)


def test_analyze_would_raise_error(dg: Graph):
    class DataBase:
        def __init__(self, engine: int):
            self.engine = engine

    class Repository:
        def __init__(self, db: DataBase):
            self.db = db

    class UserService:
        def __init__(self, repository: Repository):
            self.repository = repository

    dg.analyze(UserService)


async def test_analyze_a_factory(dg: Graph):
    class DataBase:
        def __init__(self, engine: str):
            self.engine = engine

        def close(self) -> None:
            return

    def db_factory() -> DataBase:
        return DataBase("test")

    db = dg.resolve(db_factory)
    assert db.engine == "test"

    len(dg.type_registry)

    dg.resolve(DataBase)


def test_sr_false():
    dg = Graph()

    class DataBase:
        def __init__(self, engine: str):
            self.engine = engine

        def close(self) -> None:
            return

    def db_factory() -> DataBase:
        return DataBase("test")

    dg.node(db_factory)
    dg.analyze_nodes()


def test_sr_pre_resolved():
    dg = Graph()

    class DataBase:
        def __init__(self, engine: str):
            self.engine = engine

        def close(self) -> None:
            return

    @dg.node
    def db_factory() -> DataBase:
        return DataBase("test")

    dg.analyze(db_factory)

    dg.analyze_nodes()


def test_ignore():
    dg = Graph()

    @dg.node(ignore=(0, str, "c"))
    class Item:
        def __init__(self, a: int, b: str, c: int):
            self.a = a
            self.b = b
            self.c = c

    # TypeError: __init__() missing 3 required positional arguments: 'a', 'b', and 'c'
    with pytest.raises(TypeError):
        dg.resolve(Item)

    item = dg.resolve(Item, a=1, b="2", c=3)
    assert item.a == 1 and item.b == "2" and item.c == 3
