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

    class UserRepo:
        def __init__(self, db: Database):
            self.db = db

        def test(self):
            return "test"

    class SessionRepo:
        def __init__(self, db: Database):
            self.db = db

    @dg.node(lazy=True)
    class ServiceA:
        def __init__(self, user_repo: UserRepo, session_repo: SessionRepo):
            self.user_repo = user_repo
            self.session_repo = session_repo

    instance = dg.resolve(ServiceA)
    assert isinstance(instance, ServiceA)
    assert isinstance(instance.user_repo, LazyDependent)
    assert isinstance(instance.session_repo, LazyDependent)
    assert isinstance(instance.user_repo.db, LazyDependent)
    assert isinstance(instance.session_repo.db, LazyDependent)
    assert instance.user_repo.test() == "test"
