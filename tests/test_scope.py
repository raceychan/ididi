import gc
import typing as ty
from typing import Annotated

import pytest

from ididi import Graph, Ignore, Resource, use
from ididi.config import DefaultScopeName
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
    dg = Graph()

    dg.node(async_get_client)
    dg.node(async_get_db)

    @dg.entry
    async def main(db: Annotated[AsyncDataBase, use(AsyncDataBase)]) -> str:
        assert db.is_opened
        return "ok"

    assert await main() == "ok"


def test_gen_factory():
    dg = Graph()

    dg.node(get_client)
    dg.node(get_db)

    @dg.entry
    def main(db: Annotated[DataBase, use(DataBase)]) -> str:
        assert db.is_opened
        return "ok"

    assert main() == "ok"


def test_sync_func_requires_async_factory():
    dg = Graph()

    class AsyncResource(AsyncResourceBase): ...

    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        ar = AsyncResource()
        yield ar

    dg.node(get_async_resource)

    @dg.entry
    def main(ar: Annotated[AsyncResource, use(AsyncResource)]) -> str:
        assert ar.is_opened
        return "ok"

    with pytest.raises(AsyncResourceInSyncError):
        main()


async def test_scope_repeat_resolve():
    dg = Graph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    class AsyncResource(AsyncResourceBase): ...

    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource

    dg.node(use(get_resource, reuse=True))

    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource

    dg.node(use(get_async_resource, reuse=True))

    with dg.scope() as scope:
        resource = scope.resolve(Resource)
        sec_resource = scope.resolve(Resource)
        assert resource.is_opened
        assert resource is sec_resource

    async with dg.ascope() as scope:
        resource = await scope.resolve(AsyncResource)
        sec_resource = await scope.resolve(AsyncResource)
        assert resource.is_opened
        assert resource is sec_resource


async def test_resource_shared_within_scope():
    dg = Graph()

    dg.node(use(get_db, reuse=True))
    dg.node(use(get_client, reuse=True))
    dg.node(use(async_get_client, reuse=True))
    dg.node(use(async_get_db, reuse=True))

    class FirstResource(AsyncResourceBase):
        def __init__(self, database: AsyncDataBase):
            self.database = database

    class SecondResource(AsyncResourceBase):
        def __init__(self, database: AsyncDataBase):
            self.database = database

    assert dg.should_be_scoped(FirstResource)



    async with dg.ascope() as scope:
        first_resource = await scope.resolve(FirstResource)
        second_resource = await scope.resolve(SecondResource)

        assert first_resource.database is second_resource.database
        assert first_resource.database.is_opened

    assert first_resource.database.is_closed


async def test_nested_scope():
    dg = Graph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    class AsyncResource(AsyncResourceBase): ...

    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    dg.node(get_resource)

    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource
        await resource.close()

    dg.node(get_async_resource)

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
    dg = Graph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    class AsyncResource(AsyncResourceBase): ...

    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    dg.node(use(get_resource, reuse=True))

    async def get_async_resource() -> ty.AsyncGenerator[AsyncResource, None]:
        resource = AsyncResource()
        await resource.open()
        yield resource
        await resource.close()

    dg.node(use(get_async_resource, reuse=True))

    with pytest.raises(ResourceOutsideScopeError):
        await dg.aresolve(Resource)

    async with dg.ascope():

        aresource = await dg.use_scope(as_async=True).resolve(AsyncResource)
        assert aresource.is_opened

        async def inner():
            local = await dg.use_scope(as_async=True).resolve(AsyncResource)
            assert local is aresource
            assert local.is_opened
            return local

        async def new_scope():
            async with dg.ascope() as new_scope:
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
    dg = Graph()

    class Resource(ResourceBase):
        def __init__(self):
            super().__init__()

    def get_resource() -> ty.Generator[Resource, None, None]:
        resource = Resource()
        resource.open()
        yield resource
        resource.close()

    dg.node(use(get_resource, reuse=False))

    with dg.scope() as s:
        r1 = s.resolve(Resource)
        r2 = s.resolve(Resource)
        assert r1 is not r2
        assert r1.is_opened and r2.is_opened

    assert r1.is_closed and r2.is_closed

    with pytest.raises(ResourceOutsideScopeError):
        dg.resolve(Resource)


def test_nested_scope_with_context_scope():
    dg = Graph()

    with dg.scope() as dg1:
        dg1.__enter__()
        with dg.scope() as dg2:
            with dg.scope() as dg3:
                local = dg.use_scope()
                assert dg1 is not dg2
                assert dg2 is not dg3
                assert local is dg3


async def test_async_nested_scope_with_context_scope():
    dg = Graph()

    class Normal:
        def __init__(self, name: str = "normal"):
            self.name = name

    dg.node(use(Normal, reuse=True))

    async with dg.ascope() as dg1:
        await dg1.__aenter__()
        with dg.scope() as dg2:

            normal = dg2.resolve(Normal)
            normal2 = dg2.resolve(Normal)
            assert normal.name == "normal"
            assert normal is normal2

            async with dg.ascope() as dg3:
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

    assert dg.use_scope(DefaultScopeName)


async def test_db_exec():

    dg = Graph()
    dg.node(async_get_client)
    dg.node(async_get_db)

    @dg.entry
    async def main(
        db: Annotated[AsyncDataBase, use(AsyncDataBase)], sql: Ignore[str]
    ) -> ty.Any:
        res = await db.execute(sql)
        return res

    sql = "select moeny from bank"
    assert await main(sql=sql) == sql


@pytest.mark.asyncio
async def test_scope_different_across_context():
    dg = Graph()

    class Normal:
        def __init__(self, name: str = "normal"):
            self.name = name

    dg.node(use(Normal, reuse=True))
    dg.resolve(Normal)

    with dg.scope("1") as s1:

        def func1():
            with dg.scope("2") as s2:
                with dg.scope("3") as s3:
                    repr(s1)
                    n3 = s3.resolve(Normal)
                    n1 = s1.resolve(Normal)
                    assert (
                        n1 is n3
                    ), "Non-resources reusable instances should be shared across scopes"
                    assert s3.get_scope("1") is s1
                    assert s3.get_scope("2") is s2
                    assert s3.get_scope("3") is s3

                with pytest.raises(OutOfScopeError):
                    s3 = dg.use_scope("3")

                return s1

        func1()

        def func2():
            dg.use_scope(2)

        with pytest.raises(OutOfScopeError):
            func2()


async def test_dg_create_default_scope():
    dg = Graph()

    assert dg.use_scope(DefaultScopeName)


async def test_share_single_pattern():
    dg = Graph()

    with dg.scope() as scope:
        g = scope.resolve(Graph)

        assert g is dg


async def test_scope_pre():
    dg = Graph()

    with dg.scope("s1") as s1:
        with dg.scope("s2") as s2:
            assert s2.pre is s1

    with dg.scope("n1") as n1:
        with n1.scope("n2") as n2:
            assert n2.pre is n1


async def test_scope_refcnt():
    dg = Graph()

    with dg.scope("s1") as s1:
        with dg.scope("s2") as s2:
            assert s2.pre is s1

    s1_cnt = len(gc.get_referrers(s1))
    s2_cnt = len(gc.get_referrers(s2))

    # s1 is tracked by s2 as well
    assert s1_cnt == (s2_cnt + 1)


def test_scope_gc():
    dg = Graph()

    def inner():
        with dg.scope("1") as i1:

            def inn2():
                with i1.scope("2") as i2:
                    assert i2.pre is i1

            inn2()

        i1_cnt = len(gc.get_referrers(i1))
        # i1_cnt is not referred by i2 anymore
        # which means i2 is collected.
        assert i1_cnt == 1

    inner()


async def test_ascope_ctx_exit():
    dg = Graph()

    class Conn: ...

    def get_conn() -> Resource[Conn]:
        cnn = Conn()
        yield cnn

    with pytest.raises(TypeError):
        async with dg.ascope() as asc:
            await asc.resolve(get_conn)
            raise TypeError


from typing import Annotated, Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E", bound=type[str])


def test_resolve_generic():
    class Container(Generic[T]):
        def __init__(self, name: T):
            self.name = name

    dg = Graph()

    dg.analyze(Container[str])


def test_resolve_annotated():
    class Engine: ...

    dg = Graph()

    eg = dg.resolve(Annotated[Engine, "aloha"])
    assert isinstance(eg, Engine)


def test_resolve_complex_generic():

    class EngineFactory(Generic[E]):
        def __init__(self, engine_class: E): ...

    dg = Graph()

    eg = dg.analyze(EngineFactory[type[str]])


def test_register_random_callback():
    dg = Graph()

    side_effect = []

    def my_callback(a: int, b: int):
        nonlocal side_effect
        side_effect.append(a)
        side_effect.append(b)

    with dg.scope() as scp:
        scp.register_exit_callback(my_callback, 1, 2)

    assert side_effect == [1, 2]


async def test_register_random_asynccallback():
    dg = Graph()

    side_effect = []

    async def my_callback(a: int, b: int) -> None:
        nonlocal side_effect
        side_effect.append(a)
        side_effect.append(b)

    async with dg.ascope() as scp:
        scp.register_exit_callback(my_callback, 1, 2)

    assert side_effect == [1, 2]
