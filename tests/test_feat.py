"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

from typing import Annotated

from ididi import Graph, Ignore


class User:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role


def get_user() -> Ignore[User]:
    return User("user", "admin")


def validate_admin(user: Annotated[User, get_user]) -> Ignore[str]:
    assert user.role == "admin"
    return "ok"


async def test_resolve_function():
    dg = Graph()

    user = dg.resolve(get_user)
    assert isinstance(user, User)


def test_dg_resolve_params():
    dg = Graph()

    node = dg.analyze(validate_admin)

    assert dg.resolve(validate_admin) == "ok"
