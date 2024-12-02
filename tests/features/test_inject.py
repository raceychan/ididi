# from ididi import DependencyGraph, entry, inject
# from tests.features.services import UserService


# def get_user_service() -> UserService:
#     dg = DependencyGraph()
#     return dg.resolve(UserService)


# async def create_user(service: UserService = inject(get_user_service)):
#     assert isinstance(service, UserService)


# async def test_inject_entry():
#     f = entry(create_user)
#     await f()
