import pytest

from ididi.errors import UnsolvableDependencyError
from ididi.graph import DependencyGraph


def test_improved_error():
    dg = DependencyGraph()

    class Config:
        def __init__(self, env):
            self.env = env

    @dg.node
    def config_factory() -> Config:
        return Config("dev")

    class DataBase:
        def __init__(self, config: Config):
            self.config = config

    class CacheService:
        def __init__(self, config: Config):
            self.config = config

    class AuthService:
        def __init__(self, db: DataBase, cache: CacheService):
            self.db = db
            self.cache = cache

    class UserService:
        def __init__(self, auth: AuthService, db: DataBase):
            self.auth = auth
            self.db = db

    class NotificationService:
        def __init__(self, config: Config):
            self.config = config

    class EmailService:
        def __init__(self, notification: NotificationService, user: UserService):
            self.notification = notification
            self.user = user

    email = dg.resolve(EmailService)
    assert email.notification.config.env == "dev"
