"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

from typing import Annotated

from ididi import entry, inject
from tests.test_data import UserService


def get_user_service() -> UserService:
    return UserService(1, 2)


async def deep_nested(
    service: Annotated[
        UserService,
        Annotated[UserService, Annotated[UserService, inject(get_user_service)]],
    ]
):
    assert isinstance(service, UserService)
    assert service.db == 1
    assert service.auth == 2
    return "ok"


async def test_nested_annt_entry():
    f = entry(deep_nested)
    assert await f() == "ok"
