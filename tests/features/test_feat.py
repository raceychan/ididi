"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

import typing as ty
from contextlib import asynccontextmanager

import pytest

from ididi import DependencyGraph, entry

dg = DependencyGraph()


class DataBase:
    def __init__(self, url: str = ""):
        self.url = url
        self._is_closed = False

    async def close(self) -> None:
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed


@dg.node
async def get_db() -> ty.AsyncGenerator[DataBase, None]:
    db = DataBase(url="sqlite://")
    try:
        yield db
    finally:
        await db.close()


@dg.entry_node
async def main(db: DataBase):
    assert not db.is_closed


@pytest.mark.debug
@pytest.mark.asyncio
async def test_async_gen_factory():
    await main()
    # db = dg.resolve(get_db)
    # breakpoint()

    # dg = DependencyGraph()

    # class DataBase:
    #     def __init__(self, url: str = ""):
    #         self.url = url
    #         self._is_closed = False

    #     async def close(self) -> None:
    #         self._is_closed = True

    #     @property
    #     def is_closed(self) -> bool:
    #         return self._is_closed

    # @dg.node(reuse=False)
    # @asynccontextmanager
    # async def get_db() -> ty.AsyncGenerator[DataBase, None]:
    #     db = DataBase(url="sqlite://")
    #     try:
    #         yield db
    #     finally:
    #         await db.close()

    # async def main():
    #     dg.static_resolve(get_db)
