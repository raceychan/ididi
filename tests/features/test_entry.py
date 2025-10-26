from dataclasses import dataclass
from typing import Annotated

import pytest

from ididi import Graph, Ignore, entry, resolve, use


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


def main(
    email: Annotated[EmailService, use(EmailService)],
    es: Annotated[EventStore, use(EventStore)],
) -> str:
    assert isinstance(email, EmailService)
    assert isinstance(es, EventStore)
    assert email.user.auth.db.config.env == es.db.config.env
    return "ok"


def second_entry(
    email: Annotated[EmailService, use(EmailService)], config: Config
) -> str:
    assert isinstance(email, EmailService)
    return "hello"


async def third_entry(
    notification: Annotated[NotificationService, use(NotificationService)],
    name: str = "hello",
    age: int = 18,
) -> str:
    assert isinstance(notification, NotificationService)
    return f"{name} is {age} years old"


async def test_graph_entry():
    func = entry(third_entry)
    func_main = entry(main)
    func_sec = entry(second_entry)
    assert func_main() == "ok"
    assert func_sec(config=Config()) == "hello"
    assert await func() == "hello is 18 years old"


def test_solve():
    email = resolve(EmailService)
    assert isinstance(email, EmailService)


async def test_entry_with_overrides():
    @entry
    async def func4(
        notif: Annotated[NotificationService, use(NotificationService)],
        email: Ignore[str],
        format: Ignore[str],
    ):
        return email, format

    a, b = await func4(email="asdf", format="asdf")
    assert a == "asdf"
    assert b == "asdf"


@dataclass
class CreateUser:
    user_name: str
    user_email: str


async def test_entry_with_ignore():
    @entry(ignore=(CreateUser,))
    async def func4(
        service: Annotated[NotificationService, use(EmailService)],
        cmd: Ignore[CreateUser],
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="1", user_email="2")
    r = await func4(cmd=cmd)
    assert r == "1"


async def test_dg_entry_with_override():
    dg = Graph()

    @dg.entry(ignore=(CreateUser,))
    async def func4(
        service: Annotated[NotificationService, use(NotificationService)],
        cmd: CreateUser,
        *,
        config: Config,
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="1", user_email="2")
    r = await func4(cmd=cmd, config=Config())
    assert r == "1"


class Session:
    def __init__(self, config: Config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    def __enter__(self): ...

    def __exit__(self, exc_type, exc, tb): ...


async def test_dg_entry_with_acm():
    dg = Graph()

    @dg.entry(ignore=CreateUser)
    async def func4(
        service: Annotated[NotificationService, use(NotificationService)],
        cmd: CreateUser,
        *,
        session: Annotated[Session, use(Session)],
        config: Config,
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="1", user_email="2")
    r = await func4(cmd=cmd, config=Config())
    assert r == "1"


def test_sync_entry_with_override():
    dg = Graph()

    @dg.entry(ignore=CreateUser)
    def func4(
        service: Annotated[NotificationService, use(NotificationService)],
        cmd: CreateUser,
        *,
        config: Config,
        session: Annotated[Session, use(Session)],
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="0", user_email="2")
    r = func4(cmd=cmd, config=Config())
    assert r == "0"


def test_entry_reuse():
    dg = Graph()

    rounds = 100
    dg.reset()

    def create_user(
        user_name: Ignore[str],
        user_email: Ignore[str],
        service: Annotated[UserService, use(UserService, reuse=False)],
    ) -> UserService:
        return service

    SERVICES = set[UserService]()

    create_user = dg.entry(reuse=False)(create_user)

    for _ in range(rounds):
        SERVICES.add(create_user("test", "email"))

    assert len(SERVICES) == rounds

    SERVICES.clear()

    dg.reset(clear_nodes=True)
    dg.node(use(UserService, reuse=True))

    for _ in range(rounds):
        SERVICES.add(create_user("test", "email"))

    assert len(SERVICES) == 1


async def test_entry_replace():
    dg = Graph()

    async def create_user(
        user_name: str,
        user_email: str,
        service: Annotated[UserService, use(UserService)],
    ) -> UserService:
        return service

    class FakeUserService(UserService): ...

    create_user = dg.entry(reuse=False)(create_user)
    create_user.replace(UserService, FakeUserService)

    res = await create_user("user", "user@email.com")
    assert isinstance(res, FakeUserService)

    class EvenFaker(UserService): ...

    create_user.replace(service=EvenFaker)
    create_user.replace(UserService, service=EvenFaker)

    res = await create_user("user", "user@email.com")
    assert isinstance(res, EvenFaker)


async def test_entry_override_with_factory():
    dg = Graph()

    @dg.entry
    async def create_user(
        user_name: str,
        user_email: str,
        service: Annotated[UserService, use(UserService)],
    ) -> UserService:
        return service

    class FakeUserService(UserService): ...

    @dg.node
    def _() -> UserService:
        return FakeUserService("1", "2")

    service_res = await create_user("1", "2")
    assert isinstance(service_res, FakeUserService)


async def test_entry_override_with_override():
    dg = Graph()

    @dg.entry(ignore=(0, 1))
    async def create_user(
        user_name: str,
        user_email: str,
        service: Annotated[UserService, use(UserService)],
    ) -> UserService:
        return service

    @dg.node
    def user_factory() -> UserService:
        return UserService("1", 2)

    class FakeUserService(UserService): ...

    dg.override(UserService, FakeUserService)

    service_res = await create_user("1", "2")
    assert isinstance(service_res, FakeUserService)



async def test_entry_with_default_use():
    class A: ...
    
    async def main(a: Annotated[A, use(reuse=True)]) -> None:
        ...

    graph= Graph()

    main = graph.entry(main)

    await main()