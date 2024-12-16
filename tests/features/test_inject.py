from typing import Annotated

from ididi import DependencyGraph, entry, use
from tests.test_data import UserService


def get_user_service() -> UserService:
    return UserService(1, 2)


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


class AuthService: ...


def get_auth(user: Annotated[UserService, "random"]) -> AuthService:
    return AuthService()


def test_random_annotated():
    dg = DependencyGraph()
    dg.resolve(get_auth)


class SessionService:
    def __init__(self, auth: AuthService = use(get_auth)):
        self.auth = auth


def test_class_inject():
    dg = DependencyGraph()
    dg.resolve(SessionService)
