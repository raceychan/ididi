from typing import Annotated

from ididi import Graph


class User:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role


def get_user():
    return User("user", "admin")


def validate_admin(user: Annotated[User, get_user]):
    assert user.role == "admin"
    return "ok"


# def test_dg_resolve_params():
#     dg = Graph()
#     assert dg.resolve_function(validate_admin) == "ok"
