from ididi import Graph, Resource
import pytest

# from ididi._itypes import Resource


class DataBase:
    def __enter__(self): ...

    def __exit__(self, *args): ...


def get_db() -> Resource[DataBase]:
    db = DataBase()
    yield db


class UserService: ...


def get_service(dg: Graph) -> UserService:
    return dg

def test_dg_self_inject():
    dg = Graph()
    assert dg is dg.resolve(Graph)
    # assert dg is dg.resolve(get_service)


def test_dg_register_dependent():
    dg = Graph()
    service = UserService()

    dg.register_singleton(service)
    assert dg.resolve(UserService) is service


def test_dg_register_resource_dependent():
    dg = Graph()
    db = DataBase()
    dg.node(get_db)
    with dg.scope() as scope:
        scope.register_singleton(db)
        assert scope.resolve(DataBase) is db
