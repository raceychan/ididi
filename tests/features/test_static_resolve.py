import pytest

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


# Create and register nodes
dg = DependencyGraph()


def test_static_resolve():
    node = dg.static_resolve(EmailService)
    assert len(dg.nodes) == 7
    assert len(dg.resolved_nodes) == 7

    dg.resolve(EmailService)
    assert len(dg.nodes) == 7
    assert len(dg.resolved_nodes) == 7


dg2 = DependencyGraph()


class ForwardService:
    def __init__(self, client: "ForwardClient"):
        self.client = client


class ForwardClient:
    def __init__(self, config: "ForwardConfig"):
        self.config = config


class ForwardConfig:
    pass


def test_forward_dependency():
    node = dg2.static_resolve(ForwardService)
    assert len(dg2.nodes) == 3
    assert len(dg2.resolved_nodes) == 3

    dg2.resolve(ForwardService)
    assert len(dg2.nodes) == 3
    assert len(dg2.resolved_nodes) == 3


@pytest.mark.asyncio
async def test_async_enter():
    @dg2.node(reuse=False)
    class AsyncService:
        pass

    async with dg2:
        dg2.resolve(AsyncService)
