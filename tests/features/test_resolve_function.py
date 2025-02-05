from typing import Annotated

from ididi import Graph, Ignore

from ..test_data import Config, UserService


class User:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role


def get_user(config: Config) -> Ignore[User]:
    assert isinstance(config, Config)
    return User("user", "admin")


def validate_admin(
    user: Annotated[User, get_user], service: UserService
) -> Ignore[str]:
    assert user.role == "admin"
    assert isinstance(service, UserService)
    return "ok"


async def test_resolve_function():
    dg = Graph()

    user = dg.resolve(get_user)
    assert isinstance(user, User)


def test_dg_resolve_params():
    dg = Graph()

    assert dg.resolve(validate_admin) == "ok"
