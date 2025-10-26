"""
test override dependencies
"""

from ididi import Graph


class DataBase:
    ...


class FakeDB(DataBase):
    ...


class UserRepository:
    def __init__(self, db: DataBase):
        self.db = db


def db_factory() -> DataBase:
    return FakeDB()


def test_resolve():
    dg = Graph()
    dg.node(UserRepository)
    assert isinstance(dg.resolve(UserRepository).db, DataBase)

    dg.node(db_factory)
    assert isinstance(dg.resolve(UserRepository).db, FakeDB)
