"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

# import inspect
# import typing as ty

import typing as ty

from ididi import DependencyGraph, entry, inject
from tests.test_data import UserService

t = ty.Annotated[str, "test"]


# def test_sig():
#     sig = inspect.signature(create_user)
#     annt = sig.parameters["ser"].default
#     func = annt.__args__[0]  # -> service func
#     default = annt.__metadata__[0]  #  -> "faq"


def get_user_service(dg: DependencyGraph) -> UserService:
    return dg.resolve(UserService)


async def create_user(service: UserService = inject(get_user_service)):
    assert isinstance(service, UserService)


async def test_inject_entry():
    f = entry(create_user)
    await f()
