import typing as ty

import pytest

from ididi import DependencyGraph
from ididi.errors import AsyncResourceInSyncError


class ResourceBase:
    def __init__(self):
        self._status = "init"

    def open(self) -> None:
        self._status = "opened"

    def close(self) -> None:
        self._status = "closed"

    @property
    def is_opened(self) -> bool:
        return self._status == "opened"

    @property
    def is_closed(self) -> bool:
        return self._status == "closed"


class AsyncResourceBase:
    def __init__(self):
        self._status = "init"

    async def open(self) -> None:
        self._status = "opened"

    async def close(self) -> None:
        self._status = "closed"

    @property
    def is_opened(self) -> bool:
        return self._status == "opened"

    @property
    def is_closed(self) -> bool:
        return self._status == "closed"


class Client(ResourceBase):
    def __init__(self, client_type: str = "postgres"):
        super().__init__()
        self.client_type = client_type


class DataBase(ResourceBase):
    def __init__(self, client: Client):
        super().__init__()
        self.client = client


class AsyncClient(AsyncResourceBase):
    def __init__(self, client_type: str = "postgres"):
        super().__init__()
        self.client_type = client_type


class AsyncDataBase(AsyncResourceBase):
    def __init__(self, client: AsyncClient):
        super().__init__()
        self.client = client


@pytest.mark.asyncio
async def test_async_gen_factory():
    dg = DependencyGraph()

    @dg.node
    async def get_client() -> ty.AsyncGenerator[AsyncClient, None]:
        client = AsyncClient()
        try:
            await client.open()
            yield client
        finally:
            await client.close()

    @dg.node
    async def get_db(client: AsyncClient) -> ty.AsyncGenerator[AsyncDataBase, None]:
        db = AsyncDataBase(client)
        assert client.is_opened
        try:
            await db.open()
            yield db
        finally:
            await db.close()

    @dg.entry
    async def main(db: AsyncDataBase) -> str:
        assert db.is_opened
        return "ok"

    assert await main() == "ok"


def test_gen_factory():
    dg = DependencyGraph()

    @dg.node
    def get_client() -> ty.Generator[Client, None, None]:
        client = Client()
        try:
            client.open()
            yield client
        finally:
            client.close()

    @dg.node
    def get_db(client: Client) -> ty.Generator[DataBase, None, None]:
        db = DataBase(client)
        assert client.is_opened
        try:
            db.open()
            yield db
        finally:
            db.close()

    @dg.entry
    def main(db: DataBase) -> str:
        assert db.is_opened
        return "ok"

    assert main() == "ok"


def test_sync_func_requires_async_factory():
    dg = DependencyGraph()

    class AsyncResource(AsyncResourceBase): ...

    @dg.node
    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        ar = AsyncResource()
        yield ar

    @dg.entry
    def main(ar: AsyncResource) -> str:
        assert ar.is_opened
        return "ok"

    with pytest.raises(AsyncResourceInSyncError):
        main()



@pytest.mark.asyncio
async def test_scope():
    dg = DependencyGraph()

    class Resource(ResourceBase): ...

    class AsyncResource(AsyncResourceBase): ...

    @dg.node
    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource  

    @dg.node
    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource

    with dg.scope() as scope:
        resource = scope.resolve(Resource)
        assert resource.is_opened

    async with dg.scope() as scope:
        resource = await scope.resolve(AsyncResource)
        assert resource.is_opened

