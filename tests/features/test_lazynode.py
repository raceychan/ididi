import pytest

from ididi.errors import NotSupportedError
from ididi.graph import DependencyGraph
from ididi.node import LazyDependent

"""
LazyDependent: a special kind of dependent that represents a dependency that will be resolved later.

@dg.node(lazy=True)
class UserRepository:
    def __init__(self, db: Database):
        self.db = db

Here dg.resolve(UserRepository) will return a user_repository instance with db being a LazyDependent.

"""


@pytest.mark.skip("TODO: support recursive lazy dependent")
def test_lazynode():
    dg = DependencyGraph()

    class Config:
        def __init__(self, file: str = "config.json"):
            self.file = file

    class Database:
        def __init__(self, config: Config):
            self.config = config

        def save(self, name: str):
            return f"saved {name}"

    class UserRepo:
        def __init__(self, db: Database, config: Config):
            self._db = db
            self.config = config

        def test(self):
            return "test"

        def save(self, name: str):
            return self._db.save(name)

        @property
        def db(self) -> Database:
            return self._db

    class SessionRepo:
        def __init__(self, db: Database):
            self._db = db

    @dg.node(lazy=True)
    class ServiceA:
        def __init__(self, user_repo: UserRepo, session_repo: SessionRepo):
            self._user_repo = user_repo
            self._session_repo = session_repo

            assert isinstance(self._user_repo, LazyDependent)
            assert isinstance(self._session_repo, LazyDependent)

        @property
        def user_repo(self) -> UserRepo:
            return self._user_repo

        @property
        def session_repo(self) -> SessionRepo:
            return self._session_repo

    instance = dg.resolve(ServiceA)
    assert isinstance(instance, ServiceA)

    assert isinstance(instance.user_repo, LazyDependent)
    repr(instance.user_repo.db)
    assert isinstance(instance.session_repo, LazyDependent)

    db1 = instance.user_repo._db
    db2 = instance.session_repo._db
    assert db1 is db2

    # TODO: support recursive lazy dependent
    assert isinstance(instance.user_repo.config, LazyDependent)

    assert instance.user_repo.test() == "test"
    repo1 = instance.user_repo
    assert instance.user_repo.save("test") == "saved test"
    repo2 = instance.user_repo
    assert repo1 is repo2


def test_lazyfactory():
    dg = DependencyGraph()

    @dg.node
    class Service:
        def __init__(self, name: str):
            self._name = name

    with pytest.raises(NotSupportedError):

        @dg.node(lazy=True)
        def service_factory() -> Service:
            return Service(name="test")
