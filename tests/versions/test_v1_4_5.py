from typing import Annotated

import pytest

from ididi import Graph, Ignore, Scoped, use


class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    @classmethod
    def from_email(cls, email: str) -> "User":
        return User("user", 1, email)


def test_class_method():
    dg = Graph()

    u = dg.resolve(User.from_email, email="e")
    assert u.name == "user"
    assert u.age == 1


class Config:
    def __init__(self, url: Ignore[str]):
        self.url = url


def get_config() -> Config:
    return Config("asdf")


class DB:
    def __init__(
        self, config: Annotated[Config, use(get_config)]
    ):
        self.config = config


def test_annotated_mark():
    dg = Graph()
    dg.analyze(Config)
    assert dg.nodes[Config].dependencies
    db = dg.resolve(DB)
    assert db.config.url == "asdf"


def test_dg_add_nodes():
    dg = Graph()

    class AuthService: ...

    class Conn: ...

    def auth_factory() -> AuthService:
        return AuthService()

    def conn_factory() -> Scoped[Conn]:
        conn = Conn()
        yield conn

    dg.add_nodes(
        use(DB, reuse=False),
        auth_factory,
        conn_factory,
    )
    assert len(dg.nodes) == 4


from typing import Any, NewType


class Request: ...


RequestParams = NewType("RequestParams", dict[str, Any])


async def test_resolve_request():
    dg = Graph()

    async def resolve_request(r: Request) -> RequestParams:
        return RequestParams({"a": 1})

    dg.node(resolve_request)
    await dg.resolve(resolve_request, r=Request())


def test_class_override_reuse():

    class Username:
        def __init__(self, name: str):
            self.name = name

    class User:
        def __init__(self, name: str, uname: Username):
            self.name = name
            self.uname = uname

    dg = Graph()
    user = dg.resolve(User, name="uuu")

    assert user.name == user.uname.name == "uuu"


def test_reuse_resolved():
    dg = Graph()

    def dependency(a: int) -> Ignore[int]:
        return a

    def main(a: int, b: int, c: Annotated[int, use(dependency)]) -> Ignore[float]:
        return a + b + c

    dg.resolve(main, a=1, b=2)
