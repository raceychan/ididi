import sys

import pytest

from ididi import Graph
from ididi.errors import MissingAnnotationError

dg = Graph()


class Config:
    def __init__(self, env):
        self.env = env


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
        # self.db = db


class NotificationService:
    def __init__(self, config: Config):
        self.config = config


class EmailService:
    def __init__(self, notification: NotificationService, user: UserService):
        self.notification = notification
        self.user = user


class EventStore:
    def __init__(self, db: DataBase):
        self.db = db


def test_improved_error():
    with pytest.raises(MissingAnnotationError) as e:
        email = dg.resolve(UserService)


def test_factory_override():
    dg.node(config_factory)
    dg.resolve(EmailService)
