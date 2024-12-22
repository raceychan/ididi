import typing as ty

import pytest

from ididi import DependencyGraph
from ididi.errors import (
    AsyncResourceInSyncError,
    OutOfScopeError,
    ResourceOutsideScopeError,
)


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

    async def execute(self, sql: str) -> str:
        return sql


async def async_get_client() -> ty.AsyncGenerator[AsyncClient, None]:
    client = AsyncClient()
    try:
        await client.open()
        yield client
    finally:
        await client.close()


async def async_get_db(client: AsyncClient) -> ty.AsyncGenerator[AsyncDataBase, None]:
    db = AsyncDataBase(client)
    assert client.is_opened
    try:
        await db.open()
        yield db
    finally:
        await db.close()


def get_client() -> ty.Generator[Client, None, None]:
    client = Client()
    try:
        client.open()
        yield client
    finally:
        client.close()


def get_db(client: Client) -> ty.Generator[DataBase, None, None]:
    db = DataBase(client)
    assert client.is_opened
    try:
        db.open()
        yield db
    finally:
        db.close()


async def test_async_gen_factory():
    dg = DependencyGraph()

    dg.node(async_get_client)
    dg.node(async_get_db)

    @dg.entry
    async def main(db: AsyncDataBase) -> str:
        assert db.is_opened
        return "ok"

    assert await main() == "ok"


def test_gen_factory():
    dg = DependencyGraph()

    dg.node(get_client)
    dg.node(get_db)

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


async def test_scope_repeat_resolve():
    dg = DependencyGraph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

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
        sec_resource = scope.resolve(Resource)
        assert resource.is_opened
        assert resource is sec_resource

    async with dg.scope() as scope:
        resource = await scope.resolve(AsyncResource)
        sec_resource = await scope.resolve(AsyncResource)
        assert resource.is_opened
        assert resource is sec_resource


async def test_resource_shared_within_scope():
    dg = DependencyGraph()

    dg.node(get_db)
    dg.node(get_client)
    dg.node(async_get_client)
    dg.node(async_get_db)

    class FirstResource(AsyncResourceBase):
        def __init__(self, database: AsyncDataBase):
            self.database = database

    class SecondResource(AsyncResourceBase):
        def __init__(self, database: AsyncDataBase):
            self.database = database

    assert dg.should_be_scoped(FirstResource)

    async with dg.scope() as scope:
        first_resource = await scope.resolve(FirstResource)
        second_resource = await scope.resolve(SecondResource)

        assert first_resource.database is second_resource.database
        assert first_resource.database.is_opened

    assert first_resource.database.is_closed


async def test_nested_scope():
    dg = DependencyGraph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    class AsyncResource(AsyncResourceBase): ...

    @dg.node
    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    @dg.node
    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource
        await resource.close()

    with dg.scope() as scope:
        resource = scope.resolve(Resource)

        def inner():
            with dg.scope() as new_scope:
                local = new_scope.resolve(Resource)
                assert local is not resource
                return local

        assert resource.is_opened
        local = inner()
        assert local.is_closed

    assert resource.is_closed


async def test_context_scope():
    dg = DependencyGraph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    class AsyncResource(AsyncResourceBase): ...

    @dg.node
    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    @dg.node
    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource
        await resource.close()

    with pytest.raises(ResourceOutsideScopeError):
        await dg.aresolve(Resource)

    async with dg.scope():

        aresource = await dg.use_scope(as_async=True).resolve(AsyncResource)
        assert aresource.is_opened

        async def inner():
            local = await dg.use_scope(as_async=True).resolve(AsyncResource)
            assert local is aresource
            assert local.is_opened
            return local

        async def new_scope():
            async with dg.scope() as new_scope:
                await new_scope.resolve(Resource)
                new = await new_scope.resolve(AsyncResource)
                assert local is not new
                assert new.is_opened
                return new

        local = await inner()
        new = await new_scope()
        assert local.is_opened
        assert new.is_closed

    assert aresource.is_closed
    assert local.is_closed


def test_non_reuse_resource():
    dg = DependencyGraph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    @dg.node(reuse=False)
    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    with dg.scope() as s:
        r1 = s.resolve(Resource)
        r2 = s.resolve(Resource)
        assert r1 is not r2
        assert r1.is_opened and r2.is_opened

    assert r1.is_closed and r2.is_closed

    with pytest.raises(ResourceOutsideScopeError):
        dg.resolve(Resource)


def test_nested_scope_with_context_scope():
    dg = DependencyGraph()

    with dg.scope() as dg1:
        with dg.scope() as dg2:
            with dg.scope() as dg3:
                local = dg.use_scope()
                assert dg1 is not dg2
                assert dg2 is not dg3
                assert local is dg3


async def test_async_nested_scope_with_context_scope():
    dg = DependencyGraph()

    class Normal:
        def __init__(self, name: str = "normal"):
            self.name = name

    async with dg.scope() as dg1:
        with dg.scope() as dg2:

            normal = dg2.resolve(Normal)
            normal2 = dg2.resolve(Normal)
            assert normal.name == "normal"
            assert normal is normal2

            async with dg.scope() as dg3:
                local = dg.use_scope(as_async=True)
                assert dg1 is not dg2
                assert dg2 is not dg3
                assert local is dg3

            second_local = dg.use_scope()
            assert dg2 is second_local
        assert dg.use_scope(as_async=True) is dg1

    def test_two():
        with dg.scope() as dga1:
            assert dga1 is not dg1
            assert dga1 is not dg2
            assert dga1 is not dg3
            assert dga1 is not local
            assert dga1 is not second_local

    test_two()

    with pytest.raises(OutOfScopeError):
        dg.use_scope()


async def test_db_exec():

    dg = DependencyGraph()
    dg.node(async_get_client)
    dg.node(async_get_db)

    @dg.entry
    async def main(db: AsyncDataBase, sql: str) -> ty.Any:
        res = await db.execute(sql)
        return res

    sql = "select moeny from bank"
    assert await main(sql=sql) == sql
