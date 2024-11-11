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


class ResourceBase:
    def __init__(self):
        self._is_closed = False

    async def close(self) -> None:
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed


class AsyncResourceBase:
    def __init__(self):
        self._is_closed = False

    async def close(self) -> None:
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed


class Client(AsyncResourceBase):
    def __init__(self, client_type: str = "postgres"):
        super().__init__()
        self.client_type = client_type


class DataBase(AsyncResourceBase):
    def __init__(self, client: Client):
        super().__init__()
        self.client = client


@dg.node
async def get_client() -> ty.AsyncGenerator[Client, None]:
    client = Client()
    try:
        yield client
    finally:
        await client.close()


@dg.node
async def get_db(client: Client) -> ty.AsyncGenerator[DataBase, None]:
    db = DataBase(client)
    assert not client.is_closed
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
