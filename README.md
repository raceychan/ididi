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

- No / minial changes to your existing code
- Smart injection based on type hints with strong support to `typing` module.
- Advanced scope support
- Formated and detail-rich error messages
- Highly performant, on part with nodejs and go


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


### Dependency factory 

assign which factory method to use with `ididi.use`

```python
from ididi import use
from datetime import datetime, timezone

def utc_factory() -> datetime:
    return datetime.now(timezone.utc)

UTC_DATETIME = Annotated[datetime, use(utc_factory)]

class Timer:
    def __init__(self, time: UTC_DATETIME):
        self.time = time

tmer = ididi.resolve(Timer)
assert tmer.time.tzinfo == timezone.utc
```

> [!TIP]
> **`DependencyGraph.node` accepts a wide arrange of types, such as dependent class, sync/async facotry, sync/async resource factory, with typing support.**

### Scope

Using Scope to manage resources

- **Infinite number of nested scope**
- **Parent scope can be accssed by its child scopes(within the same context)**
- **Resources will be shared across dependents only withint the same scope**
- **Resources will be automatically closed and destroyed when the scope is exited.**
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
> `dg.node` will leave your class/factory untouched, i.e., you can use it just like it is not decorated.

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


### More

For more detailed information, check out [Documentation](https://raceychan.github.io/ididi)

- Tutorial

- Usage of factory

- Visualize the dependency graph(beta)

- Circular Dependency Detection

- Lazy Dependency(Beta)

- Error context
