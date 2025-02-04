# Ididi

## Genius simplicity, unmatched power

**ididi is *100%* test covered and strictly typed.**

[![codecov](https://codecov.io/github/raceychan/ididi/graph/badge.svg?token=8N1AYWWN5N)](https://codecov.io/github/raceychan/ididi)
[![PyPI version](https://badge.fury.io/py/ididi.svg)](https://badge.fury.io/py/ididi)
[![Python Version](https://img.shields.io/pypi/pyversions/ididi.svg)](https://pypi.org/project/ididi/)
[![License](https://img.shields.io/github/license/raceychan/ididi)](https://github.com/raceychan/ididi/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/ididi.svg)](https://pypistats.org/packages/ididi)

---

**Documentation**: <a href="https://raceychan.github.io/ididi" target="_blank"> https://raceychan.github.io/ididi </a>

**Source Code**: <a href=" https://github.com/raceychan/ididi" target="_blank">  https://github.com/raceychan/ididi</a>

---

## Install

ididi requires python >= 3.9

```bash
pip install ididi
```

To view viusal dependency graph, install `graphviz`

```bash
pip install ididi[graphviz]
```

## Features
- **Performant**: almost as fast as hard-coded factories, one of the fastest dependency injection framework available.
- **Noninvasive**: No / minial changes to your existing code
- **Smart**: inject dependency based on type hints, with strong support to `typing` module.
- **Powerful**: Ididi does what others do, and also provides features that others don't.
- **Correct**, strictly typed, well-organized exceptions, well-formatted and detail-rich error messages


## Usage

### Quick Start

```python
from typing import AsyncGenerator
from ididi import use, entry

async def conn_factory(engine: AsyncEngine) -> AsyncGenerator[AsyncConnection, None]:
    async with engine.begin() as conn:
        yield conn

class UnitOfWork:
    def __init__(self, conn: AsyncConnection=use(conn_factory)):
        self._conn = conn

@entry
async def main(command: CreateUser, uow: UnitOfWork):
    await uow.execute(build_query(command))

# note uow is automatically injected here
await main(CreateUser(name='user'))
```

To resolve `AsyncConnection` outside of entry function

```python
from ididi import Graph

dg = Graph()

async with dg.scope():
    conn = await scope.resolve(conn_factory)
```

## Key Concepts

### Marks

ididi provides two `mark`s, `use` and `Ignore`, they are convenient shortcut for `Graph.node`.
Technically, they are just metadata carried by `typing.Annotated`, and should work fine with other Annotated metadata.

#### `use`

You can use `Graph.node` to register a dependent with its factory,
here we register dependent `Database` with its factory `db_factory`.
This means whenver we call `dg.resolve(Database)`, `db_factory` will be call.

```python
def db_factory() -> Database:
    return Database()

dg = Graph()
dg.node(db_factory)
```

Alternatively, you can annotate it inside `__init__`, this allow you to instantiate
`Graph` in a lazy manner.

```python
from ididi import use
class Repository:
    def __init__(self, db: Annotated[Database, use(db_factory)]):
        ...
```

#### `Ignore`

ididi takes a "resolve by default" approach, for dependencies you would like ididi to ignore, you can config ididi to ignore them.

- Ignore at Graph level

```python
from datetime import datetime
from pathlib import Path

dg = Graph(ignore=(datetime, Path))
```

- Ignore at Node level

```python
dg = Graph()
class Clock:
    def __init__(self, dt: datetime): ...

dg.node(Clock, ignore=datetime)
```

Alternatively, you can mark a dependency using `ididi.Ignore`,
```python
from ididi import Ignore

class Clock:
    def __init__(self, dt: Ignore[datetime]): ...
```

### Dependency factory 

```python
from ididi import use
from typing import NewType
from datetime import datetime, timezone

UserID = NewType("UserID", str)

def utc_factory() -> datetime:
    return datetime.now(timezone.utc)

def user_id_factory() -> UserID:
    return UserID(str(uuid4()))

class User:
    def __init__(self, user_id: UserID, created_at: Annotated[datetime, use(utc_factory)]):
        self.user_id = user_id
        self.created_at = created_at

user = ididi.resolve(User)
assert user.created_at.tzinfo == timezone.utc
```

> [!TIP]
> **`Graph.node` accepts a wide arrange of types, such as dependent class, sync/async facotry, sync/async resource factory, with typing support.**

### TypingSupport

ididi has strong support to `typing` module, includes:

- Optional
- Union
- Annotated
- Literal
- NewType
- TypedDict

...and more.

Check out `tests/features/test_typing_support.py` for examples.

### Scope

`Scope` is a temporary view of the graph specialized for handling resources. 

In a nutshell:

- Scope can access(read-only) registered singletons and resolved instances of its parent graph
- Dependents registered/resolved in a scope will stay in the scope.
- its parent graph can't access its registered singletons and resolved resources.

#### Using Scope to manage resources

- **Infinite number of nested scope**
- **Parent scope can be accssed by its child scopes(within the same context)**
- **Resources will be shared across dependents only withint the same scope**
- **Resources will be automatically closed when the scope is exited.**
- **Classes that implment `contextlib.AbstractContextManager` or `contextlib.AbstractAsyncContextManager` are also considered to be resources and can/should be resolved within scope.**
- **Scopes are separated by context**

> [!TIP]
If you have two call stack of `a1 -> b1` and `a2 -> b2`,
    Here `a1` and `a2` are two calls to the same function `a`,
    then, in `b1`, you can only access scope created by the `a1`, not `a2`.

This is particularly useful when you try to separate resources by route, endpoint, request, etc.

#### Async scope

```python
@dg.node
def get_resource() -> ty.Generator[Resource, None, None]:
    res = Resource()
    with res:
        yield res

@dg.node
async def get_asyncresource() -> ty.Generator[AsyncResource, None, None]:
    res = AsyncResource()
    async with res:
        yield res


with dg.scope() as scope:
    resource = scope.resolve(Resource)

# For async generator
async with dg.scope() as scope:
    resource = await scope.resolve(AsyncResource)
```

> [!TIP]
> `dg.node` will leave your class/factory untouched, i.e., you can use it as a function.
> e.g. `dg.node(get_resource, reuse=False)`

#### Contexted Scope

You can use dg.use_scope to retrive most recent scope, context-wise, this allows your to have
access the scope without passing it around, e.g.

```python
async def service_factory():
    async with dg.scope() as scope:
        service = scope.resolve(Service)
        yield service

@app.get("users")
async def get_user(service: Service = Depends(service_factory))
    await service.create_user(...)
```

Then somewhere deep in your service.create_user call stack

```python
async def create_and_publish():
    uow = dg.use_scope().resolve(UnitOfWork)
    async with uow.trans():
        user_repo.add_user(user)
        event_store.add(user_created_event)
```

Here `dg.use_scope()` would return the same scope you created in your `service_factory`.

#### Named Scope

You can create infinite level of scopes by assigning hashable name to scopes

```python
# at the top most entry of a request
async with dg.scope(request_id) as scope:
    ...
```

now scope with name `request_id` is accessible everywhere within the request context

```python
request_scope = dg.use_scope(request_id)
```

> [!NOTE]
Two or more scopes with the same name would follow most recent rule.

#### Nested Nmaed Scope

```python
async with dg.scope(app_name) as app_scope:
    async with dg.scope(router_name) as router_scope:
        async with dg.scope(endpoint_name) as endpoint_scope:
            async with dg.scope(user_id) as user_scope:
                async with dg.scope(request_id) as request_scope:
                    ...
```

For any functions called within the request_scope, you can get the most recent scope with `dg.use_scope()`,
or its parent scopes, i.e. `dg.use_scope(app_name)` to get app_scope.

## Testing

### Override class dependency resolution

You can control how ididi resolve a dependency during testing, by register the test double of the dependency using


- `Graph.override`
- `entry.replace`

Example:
For the following dependent

```py
class UserRepository:
    def __init__(self, db: DataBase):
        self.db=db

dg = Graph()
assert isinstance(dg.resolve(UserRepository).db, DataBase)
```

in you test file,

```py
class FakeDB(DataBase): ...

def db_factory() -> DataBase:
    return FakeDB()


def test_resolve():
    dg = Graph()
    assert isinstance(dg.resolve(db_factory).db, DataBase)

    dg.override(DataBase, db_factory)
    assert isinstance(dg.resolve(UserRepository).db, FakeDB)
```

Use `Graph.override` to replace `DataBase` with its test double.

### Override entry dependency resolution

```py
async def test_entry_replace():
    @ididi.entry
    async def create_user(
        user_name: str, user_email: str, service: UserService
    ) -> UserService:
        return service

    class FakeUserService(UserService): ...

    create_user.replace(UserService, FakeUserService)

    res = await create_user("user", "user@email.com")
    assert isinstance(res, FakeUserService)
```

Use `entryfunc.replace` to replace a dependency with its test double. 

#### `Graph.override` vs `entry.replace`

`Graph.override` applies to the whole graph, `entry.replace` applies to only the entry function.  


## More

For more detailed information, check out [Documentation](https://raceychan.github.io/ididi)

- Tutorial

- Usage of factory

- Visualize the dependency graph

- Circular Dependency Detection

- Error context
