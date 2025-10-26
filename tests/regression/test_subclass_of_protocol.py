import inspect
import typing as ty

import pytest

from ididi import Annotated, Graph, use


def test_subclass_of_protocol():
    dg = Graph()

    class Uow: ...

    class IEntity(ty.Protocol): ...

    class ChatSession(IEntity): ...

    E = ty.TypeVar("E", bound=IEntity)

    class IRepository(ty.Protocol[E]): ...

    class ISessionRepository(IRepository[ChatSession]): ...

    class SessionRepository(ISessionRepository):
        def __init__(self, uow: Uow):
            self._uow = uow

    repo = dg.resolve(SessionRepository)
    assert isinstance(repo, SessionRepository)


def test_double_protocol():
    dg = Graph()

    class UserRepo(ty.Protocol): ...

    class SessionRepo(ty.Protocol): ...

    class BothRepo(UserRepo, SessionRepo):
        def __init__(self, name: str = "test"):
            self.name = name

    class UserApp:
        def __init__(self, repo: UserRepo):
            self.repo = repo

    class SessionApp:
        def __init__(self, repo: SessionRepo):
            self.repo = repo


    dg.node(use(BothRepo, reuse=True))

    with dg.scope() as scope:
        user = scope.resolve(UserApp)
        session = scope.resolve(SessionApp)
        assert user.repo is session.repo

    dg = Graph()
    dg.node(use(BothRepo, reuse=True))
    user = dg.resolve(UserApp)
    session = dg.resolve(SessionApp)
    assert user.repo is session.repo
