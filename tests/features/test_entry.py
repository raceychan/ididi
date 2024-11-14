import time

import pytest

from ididi import DependencyGraph, entry, solve


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


@entry
async def third_entry(
    notification: NotificationService, name: str = "hello", age: int = 18
) -> str:
    assert isinstance(notification, NotificationService)
    return f"{name} is {age} years old"


@pytest.mark.asyncio
async def test_graph_entry():
    assert main() == "ok"
    assert second_entry() == "hello"
    assert await third_entry() == "hello is 18 years old"


@pytest.mark.benchmark(group="graph-entry")
def test_performance():
    dg = DependencyGraph()

    def hard_coded_factory():
        config = Config()
        email = EmailService(
            notification=NotificationService(config=config),
            user=UserService(
                auth=AuthService(
                    db=DataBase(config=config), cache=CacheService(config=config)
                ),
                db=DataBase(config=config),
            ),
        )
        return email

    times = int((10**2))
    start = time.perf_counter()
    for _ in range(times):
        hard_coded_factory()
    end = time.perf_counter()
    print(f"Time taken: {round(end - start, 6)} seconds")
    time1 = end - start

    static_pre = time.perf_counter()
    dg.static_resolve(EmailService)
    static_aft = time.perf_counter()
    static_cost = round(static_aft - static_pre, 6)
    print(f"static resolve cost {static_cost} seconds")

    start = time.perf_counter()
    for _ in range(times):
        dg.resolve(EmailService)
    end = time.perf_counter()
    time2 = end - start
    print(f"Time taken: {round(end - start, 6)} seconds")
    print(f"hardcoded: {round(time2 / time1, 2)} times faster")


def test_solve():
    email = solve(EmailService)
    assert isinstance(email, EmailService)
