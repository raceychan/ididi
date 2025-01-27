from typing import Literal, Optional, TypedDict

from typing_extensions import Unpack

from ididi import Graph


class DataBase: ...


class Cache: ...


class Config: ...


class KWARGS(TypedDict):
    db: DataBase
    cache: Cache


class Extra(KWARGS):
    config: Config


class Repo:
    def __init__(self, **kwargs: Unpack[KWARGS]) -> None:
        self.db = kwargs["db"]
        self.cache = kwargs["cache"]


class SubRepo(Repo):
    def __init__(self, **kwargs: Unpack[Extra]):
        super().__init__(db=kwargs["db"], cache=kwargs["cache"])
        self.config = kwargs["config"]


def test_resolve_unpack():
    dg = Graph()
    repo = dg.resolve(Repo)
    assert isinstance(repo.db, DataBase)
    assert isinstance(repo.cache, Cache)
    subrepo = dg.resolve(SubRepo)
    assert isinstance(subrepo.config, Config)


def test_resolve_literal():
    dg = Graph()

    class User:
        def __init__(self, name: Literal["user"] = "user"):
            self.name = name

    u = dg.resolve(User)
    assert u.name == "user"


def test_optional():
    dg = Graph()

    class UserRepo:
        def __init__(self, db: Optional[DataBase]):
            self.db = db

    assert isinstance(dg.resolve(UserRepo).db, DataBase)


# def test_annt():
#     dg = Graph()

#     class User: ...


#     NamedUser = Annotated[User, "NamedUser"]

#     d = dg.resolve(NamedUser)
#     breakpoint()
