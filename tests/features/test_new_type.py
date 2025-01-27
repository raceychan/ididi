from typing import NewType
from uuid import uuid4

from ididi import Graph, use

dg = Graph()


UUID = NewType("UUID", str)
AGE = NewType("AGE", int)


def uuid_factory() -> UUID:
    return UUID(str(uuid4()))


@dg.node
def age_factory() -> AGE:
    return 7


class User:
    def __init__(
        self,
        age: AGE,
        uuid: UUID = use(uuid_factory),
    ):
        self.age = age
        self.uuid = uuid


def test_uuid_factory():
    u = dg.resolve(uuid_factory)
    assert isinstance(u, str)


def test_resolve_user():
    user = dg.resolve(User)
    assert isinstance(user, User)
    assert isinstance(user.uuid, str)
    assert user.age == 7


def test_resolve_age():
    a = dg.resolve(AGE)
    assert a == 7


CommandContext = NewType("CommandContext", dict[str, str])


def command_context_factory() -> CommandContext:
    return CommandContext({"name": "test"})


def test_command_context():
    ctx = dg.resolve(command_context_factory)
    assert ctx == {"name": "test"}


def test_direct_resolve_nt():
    from datetime import datetime, timezone

    UTCDatetime = NewType("UTCDatetime", datetime)

    def utc_factory() -> UTCDatetime:
        return UTCDatetime(datetime.now(timezone.utc))

    dt = dg.resolve(utc_factory)
    assert isinstance(dt, datetime)
    assert dt.tzinfo == timezone.utc

    dt2 = dg.resolve(UTCDatetime)
    assert isinstance(dt2, datetime)
    assert dt != dt2
