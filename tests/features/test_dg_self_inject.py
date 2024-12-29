from ididi import DependencyGraph, Resource
# from ididi._itypes import Resource


class DataBase:
    def __enter__(self): ...

    def __exit__(self, *args): ...


def get_db() -> Resource[DataBase]:
    db = DataBase()
    yield db


class UserService: ...


def get_service(dg: DependencyGraph) -> UserService:
    return dg


def test_dg_self_inject():
    dg = DependencyGraph()
    assert dg is dg.resolve(get_service)


def test_dg_register_dependent():
    dg = DependencyGraph()
    service = UserService()

    dg.register_singleton(service, UserService)
    assert dg.resolve(UserService) is service


def test_dg_register_resource_dependent():
    dg = DependencyGraph()
    db = DataBase()
    dg.node(get_db)
    with dg.scope() as scope:
        scope.register_dependent(db, DataBase)
        assert scope.resolve(DataBase) is db
