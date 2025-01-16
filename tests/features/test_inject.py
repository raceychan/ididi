from datetime import datetime, timezone
from typing import Annotated

from ididi import DependencyGraph, entry, use
from tests.test_data import UserService


def get_user_service() -> UserService:
    return UserService(db=1, auth=2)  # type: ignore


async def create_user(service: UserService = use(get_user_service)):
    assert isinstance(service, UserService)
    assert service.db == 1
    assert service.auth == 2
    return "ok"


async def test_inject_entry():
    f = entry(create_user)
    assert await f() == "ok"


async def deep_nested(
    service: Annotated[
        UserService,
        "something",
        Annotated[str, "random", Annotated[UserService, use(get_user_service)]],
    ]
):
    assert isinstance(service, UserService)
    assert service.db == 1
    assert service.auth == 2
    return "aloha"


async def test_nested_annt_entry():
    f = entry(deep_nested)
    assert await f() == "aloha"


def utc_factory() -> datetime:
    return datetime.now(timezone.utc)


UTC_DATETIME = Annotated[datetime, use(utc_factory)]


def test_resolve_timer():
    class Timer:
        def __init__(self, time: UTC_DATETIME):
            self.time = time

    dg = DependencyGraph()

    dg.analyze(Timer)
    assert dg.nodes[datetime].factory is utc_factory
    tmer = dg.resolve(Timer)
    assert tmer.time.tzinfo == timezone.utc


def test_plain_annotated():

    class Clock:
        def __init__(self, time: Annotated[datetime, "aloha"]):
            self._time = time

    dg = DependencyGraph()
    dg.analyze(Clock)


class A:
    def __init__(self, name: str, time: UTC_DATETIME):
        self.name = name
        self.time = time


def a_f(time: UTC_DATETIME) -> A:
    return A("test", time)


class B:
    def __init__(self, *, a: A = use(a_f), age: int = 7):
        self.alice = a
        self.age = age


def b_f(a: A = use(a_f)) -> B:
    return B(a=a, age=5)


class C:
    def __init__(self, b: B = use(b_f)):
        self.berry = b


def test_nested_inject():
    dg = DependencyGraph()
    cream = dg.resolve(C)

    assert cream.berry.age == 5
    assert cream.berry.alice.name == "test"
    assert cream.berry.alice.time


def test_self_inject():
    dg = DependencyGraph()

    def user_factory(dg: DependencyGraph) -> DependencyGraph:
        return dg

    g = dg.resolve(user_factory)
    assert g is dg

    dg.remove_singleton(DependencyGraph)
