import ididi
from tests.features.services import UserService


def test_ididi_solve():
    assert isinstance(ididi.resolve(UserService), UserService)


# @ididi.entry
# def main(user_service: UserService):
#     assert isinstance(user_service, UserService)
#     return user_service


# def test_ididi_entry():
#     assert isinstance(main(), UserService)
