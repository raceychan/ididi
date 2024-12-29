import pytest

from ididi.errors import ForwardReferenceNotFoundError, UnsolvableDependencyError
from ididi.graph import DependencyGraph

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
def dg() -> DependencyGraph:
    return DependencyGraph()


def test_static_resolve(dg: DependencyGraph):
    _ = dg.static_resolve(EmailService)
    assert len(dg.nodes) == 7
    assert len(dg.resolved_nodes) == 7


def test_static_resolve_equal_resolve(dg: DependencyGraph):
    _ = dg.static_resolve(EmailService)
    assert len(dg.nodes) == 7
    assert len(dg.resolved_nodes) == 7

    dg.reset()

    dg.resolve(EmailService)

    assert len(dg.nodes) == 7
    assert len(dg.resolved_nodes) == 7


class ForwardService:
    def __init__(self, client: "ForwardClient"):
        self.client = client


class ForwardClient:
    def __init__(self, config: "ForwardConfig"):
        self.config = config


class ForwardConfig:
    pass


def test_forward_dependency(dg: DependencyGraph):
    _ = dg.static_resolve(ForwardService)
    assert len(dg.nodes) == 3
    assert len(dg.resolved_nodes) == 3

    dg.resolve(ForwardService)
    assert len(dg.nodes) == 3
    assert len(dg.resolved_nodes) == 3


@pytest.mark.asyncio
async def test_async_enter(dg: DependencyGraph):
    @dg.node(reuse=False)
    class AsyncService:
        pass

    dg.resolve(AsyncService)


def test_forward_ref_in_local_scope():
    dag = DependencyGraph()

    class ServiceA:
        def __init__(self, b: "ServiceB"):
            self.b = b

    class ServiceB:
        def __init__(self, a: str = "a"):
            self.a = a

    with pytest.raises(ForwardReferenceNotFoundError):
        dag.static_resolve(ServiceA)


def test_static_resolve_would_raise_error(dg: DependencyGraph):
    class DataBase:
        def __init__(self, engine: int):
            self.engine = engine

    class Repository:
        def __init__(self, db: DataBase):
            self.db = db

    class UserService:
        def __init__(self, repository: Repository):
            self.repository = repository

    with pytest.raises(UnsolvableDependencyError):
        dg.static_resolve(UserService)


async def test_static_resolve_a_factory(dg: DependencyGraph):
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
    dg = DependencyGraph()

    class DataBase:
        def __init__(self, engine: str):
            self.engine = engine

        def close(self) -> None:
            return

    @dg.node
    def db_factory() -> DataBase:
        return DataBase("test")

    dg.static_resolve_all()


def test_sr_pre_resolved():
    dg = DependencyGraph()

    class DataBase:
        def __init__(self, engine: str):
            self.engine = engine

        def close(self) -> None:
            return

    @dg.node
    def db_factory() -> DataBase:
        return DataBase("test")

    dg.static_resolve(db_factory)

    dg.static_resolve_all()
