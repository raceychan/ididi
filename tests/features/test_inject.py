from datetime import datetime, timezone
from typing import Annotated, NewType


from ididi import Graph, entry, use
from tests.test_data import UserService


def get_user_service() -> UserService:
    return UserService(db=1, auth=2)  # type: ignore


async def create_user(service: Annotated[UserService , use(get_user_service)]):
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


UTC_DATETIME = NewType("UTC_DATETIME", datetime)


def utc_factory() -> UTC_DATETIME:
    return UTC_DATETIME(datetime.now(timezone.utc))


def test_resolve_timer():
    class Timer:
        def __init__(self, time: Annotated[UTC_DATETIME, use(utc_factory)]):
            self.time = time

    dg = Graph()

    dg.analyze(Timer)
    node = dg.search("UTC_DATETIME")
    assert node and node.factory is utc_factory
    tmer = dg.resolve(Timer)
    assert tmer.time.tzinfo == timezone.utc


def test_plain_annotated():
    class Clock:
        def __init__(self, time: Annotated[datetime, "aloha"]):
            self._time = time

    dg = Graph()
    dg.analyze(Clock)


class A:
    def __init__(self, name: str, time: UTC_DATETIME):
        self.name = name
        self.time = time


def a_f(time: UTC_DATETIME) -> A:
    return A("test", time)


class B:
    def __init__(self, *, a: Annotated[A, use(a_f)], age: int = 7):
        self.alice = a
        self.age = age


def b_f(a: Annotated[A , use(a_f)]) -> B:
    return B(a=a, age=5)


class C:
    def __init__(self, b: Annotated[B , use(b_f)]):
        self.berry = b


def test_nested_inject():
    dg = Graph()
    cream = dg.resolve(C)

    assert cream.berry.age == 5
    assert cream.berry.alice.name == "test"
    assert cream.berry.alice.time


def test_self_inject():
    dg = Graph()

    def graph_factory(dg: Graph) -> Graph:
        return dg

    g = dg.resolve(graph_factory)
    assert g is dg

    dg.remove_singleton(Graph)
