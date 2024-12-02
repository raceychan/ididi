from ididi import DependencyGraph, entry, inject
from tests.test_data import UserService


def get_user_service(dg: DependencyGraph) -> UserService:
    return UserService(1, 2)


async def create_user(service: UserService = inject(get_user_service)):
    assert isinstance(service, UserService)
    assert service.db == 1
    assert service.auth == 2
    return "ok"


async def test_inject_entry():
    f = entry(create_user)
    assert await f() == "ok"
