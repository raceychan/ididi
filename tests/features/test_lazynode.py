import pytest

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


@pytest.mark.debug
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
        def __init__(self, db: Database):
            self._db = db

        def test(self):
            return "test"

        def save(self, name: str):
            return self._db.save(name)

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
    assert isinstance(instance.session_repo, LazyDependent)

    db1 = instance.user_repo._db
    db2 = instance.session_repo._db
    assert db1 is db2

    assert instance.user_repo.test() == "test"
    repo1 = instance.user_repo
    assert instance.user_repo.save("test") == "saved test"
    repo2 = instance.user_repo
    assert repo1 is repo2
