import typing as ty
from contextlib import asynccontextmanager

import pytest

from ididi import DependencyGraph, entry




@pytest.mark.debug
@pytest.mark.asyncio
async def test_async_gen_factory():
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

    @dg.node(reuse=False)
    @asynccontextmanager
    async def get_db() -> ty.AsyncGenerator[DataBase, None]:
        db = DataBase(url="sqlite://")
        try:
            yield db
        finally:
            await db.close()

    async def main():
        dg.static_resolve(get_db)

    # @dg.entry_node
    # async def main(db: DataBase):
    #     print(db)
    #     assert not db.is_closed
    #     service = entry(main)
    #     assert isinstance(service, main)
    #     assert not service.db.is_closed

    # await main()
