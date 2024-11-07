import pytest

from ididi import entry


class Config:
    def __init__(self, env: str = "test"):
        self.env = env


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


class EventStore:
    def __init__(self, db: DataBase):
        self.db = db


@entry
def main(email: EmailService, es: EventStore) -> str:
    assert isinstance(email, EmailService)
    assert isinstance(es, EventStore)
    assert email.user.auth.db.config.env == es.db.config.env
    return "ok"


@entry
def second_entry(email: EmailService) -> str:
    assert isinstance(email, EmailService)
    return "hello"


@pytest.mark.debug
def test_graph_entry():
    assert main() == "ok"
    assert second_entry() == "hello"