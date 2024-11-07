import inspect
import typing as ty

import pytest

from ididi import DependencyGraph


def test_subclass_of_protocol():
    dg = DependencyGraph()

    class Uow: ...

    class IEntity(ty.Protocol): ...

    class ChatSession(IEntity): ...

    class IRepository[TEntity: IEntity](ty.Protocol): ...

    class ISessionRepository(IRepository[ChatSession]): ...

    class SessionRepository(ISessionRepository):
        def __init__(self, uow: Uow):
            self._uow = uow

    repo = dg.resolve(SessionRepository)
    assert isinstance(repo, SessionRepository)
