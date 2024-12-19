from dataclasses import dataclass

from ididi import DependencyGraph, entry, resolve


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


def main(email: EmailService, es: EventStore) -> str:
    assert isinstance(email, EmailService)
    assert isinstance(es, EventStore)
    assert email.user.auth.db.config.env == es.db.config.env
    return "ok"


def second_entry(email: EmailService, config: Config) -> str:
    assert isinstance(email, EmailService)
    return "hello"


async def third_entry(
    notification: NotificationService, name: str = "hello", age: int = 18
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
    async def func4(notif: NotificationService, email: str, format: str):
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
    async def func4(service: NotificationService, cmd: CreateUser) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="1", user_email="2")
    r = await func4(cmd=cmd)
    assert r == "1"


async def test_dg_entry_with_override():
    dg = DependencyGraph()

    @dg.entry(ignore=(CreateUser,))
    async def func4(
        service: NotificationService, cmd: CreateUser, *, config: Config
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
    dg = DependencyGraph()

    @dg.entry(ignore=(CreateUser,))
    async def func4(
        service: NotificationService,
        cmd: CreateUser,
        *,
        session: Session,
        config: Config,
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="1", user_email="2")
    r = await func4(cmd=cmd, config=Config())
    assert r == "1"


def test_sync_entry_with_override():

    dg = DependencyGraph()

    @dg.entry(ignore=(CreateUser,))
    def func4(
        service: NotificationService,
        cmd: CreateUser,
        *,
        config: Config,
        session: Session,
    ) -> str:
        return cmd.user_name

    cmd = CreateUser(user_name="0", user_email="2")
    r = func4(cmd=cmd, config=Config())
    assert r == "0"
