
# Ididi
[![codecov](https://codecov.io/github/raceychan/ididi/graph/badge.svg?token=8N1AYWWN5N)](https://codecov.io/github/raceychan/ididi)
[![PyPI version](https://badge.fury.io/py/ididi.svg)](https://badge.fury.io/py/ididi)
[![Python Version](https://img.shields.io/pypi/pyversions/ididi.svg)](https://pypi.org/project/ididi/)
[![License](https://img.shields.io/github/license/raceychan/ididi)](https://github.com/raceychan/ididi/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/ididi.svg)](https://pypistats.org/packages/ididi)


## A type-based, full-fledged dependency injection library.

**ididi is *100%* test covered and strictly typed.**


📚 Docs: : <a href="https://raceychan.github.io/ididi" target="_blank"> https://raceychan.github.io/ididi </a>

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
- **Powerful**: Ididi does what alternatives do, and also provides features that others don't.
- **Performant**: almost as fast as hard-coded factories, one of the fastest dependency injection framework available.
- **Noninvasive**: No / minial changes to your existing code
- **Smart**: inject dependency based on type hints, with strong support to `typing` module.
- **Correct**, strictly typed, well-organized exceptions, well-formatted and detail-rich error messages

### TypingSupport

ididi has strong support to `typing` module, includes:

- TypedDict
- Unpack
- NewType
- Annotated
- Literal
- Optional
- Union

...and more.

Check out `tests/features/test_typing_support.py` for examples.

#### Choosing reuse strategy for typed dependencies

- Dependency type used as singleton → `reuse=True`
- Dependency type with limited instances → wrap the target with `typing.NewType`
- Dependency used as transient → `reuse=False`

#### Example: solving limited agent types with `NewType`

When only a handful of instances of the same type should be resolvable, wrap each variant in a distinct `NewType`. This keeps the factories strongly typed without turning everything into a singleton.

```python
from typing import NewType
from ididi import Graph

class LLMService: ...

class Agent:
    def __init__(self, prompt: str, llm: LLMService):
        self.prompt = prompt
        self.llm = llm

MathAgent = NewType("MathAgent", Agent)
EnglishAgent = NewType("EnglishAgent", Agent)

def math_agent_factory(llm: LLMService) -> MathAgent:
    return MathAgent(Agent(prompt="Solve math problems", llm=llm))

def english_agent_factory(llm: LLMService) -> EnglishAgent:
    return EnglishAgent(Agent(prompt="Assist with English tasks", llm=llm))

dg = Graph()
math_agent = dg.resolve(math_agent_factory)
assert isinstance(math_agent, Agent) and math_agent.prompt == "Solve math problems"
english_agent = dg.resolve(english_agent_factory)
assert isinstance(english_agent, Agent) and english_agent.prompt == "Assist with English tasks"
```

See `tests/features/test_typing_support.py::test_solve_agent_new_types` for the full test case.

## Usage

### Quick Start

```python
from typing import AsyncGenerator
from ididi import use, entry

async def conn_factory(engine: AsyncEngine) -> AsyncGenerator[AsyncConnection, None]:
    async with engine.begin() as conn:
        yield conn

class UnitOfWork:
    def __init__(self, conn: Annotated[AsyncConnection, use(conn_factory)]):
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

async with dg.ascope():
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

For dependencies you would like ididi to ignore, you can config ididi to ignore them.

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

### Function dependency

Declear a function as a dependency by using `Ignore` to annotate its return type.

```python
@dataclass
class User:
    name: str
    role: str

def get_user(config: Config) -> Ignore[User]:
    assert isinstance(config, Config)
    return User("user", "admin")

def validate_admin(
    user: Annotated[User, use(get_user)], service: UserService
) -> Ignore[str]:
    assert user.role == "admin"
    assert isinstance(service, UserService)
    return "ok"

class Route:
    def __init__(self, validte_permission: Annotated[str, use(validate_admin)]):
        assert validte_permission == "ok"

assert dg.resolve(validate_admin) == "ok"
assert isinstance(dg.resolve(Route), Route)
```

Since `get_user` returns `Ignore[User]` instead of `User`, it won't be used as factory to resolve `User`.

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
async with dg.ascope() as scope:
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
    async with dg.ascope() as scope:
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
async with dg.ascope(request_id) as scope:
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
async with dg.ascope(app_name) as app_scope:
    async with dg.ascope(router_name) as router_scope:
        async with dg.ascope(endpoint_name) as endpoint_scope:
            async with dg.ascope(user_id) as user_scope:
                async with dg.ascope(request_id) as request_scope:
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


### CheatSheet

This cheatsheet is designed to give you a quick glance at some of the basic usages of ididi.

```python
from ididi import Graph, Resolver, Ignore

class Base:
    def __init__(self, source: str = "class"):
        self.source = source

class CTX(Base): 
    def __init__(self, source: str="class"): 
        super().__init__(source)
        self.status = "init"
    async def __aenter__(self):
        self.status = "started" 
        return self
    async def __aexit__(self, *args):
        self.status = "closed"

class Engine(Base): ...

class Connection(CTX): 
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine

def get_engine() -> Engine:
    return Engine("factory")

async def get_conn(engine: Engine) -> Connection:
    async with Connection(engine) as conn:
        yield conn

async def func_dep(engine: Engine, conn: Connection) -> Ignore[int]:
    return 69

async def test_ididi_cheatsheet():
    dg = Graph()
    assert isinstance(dg, Resolver)

    engine = dg.resolve(Engine)  # resolve a class
    assert isinstance(engine, Engine) and engine.source == "class"

    faq_engine = dg.resolve(get_engine)  # resolve a factory function of a class
    assert isinstance(faq_engine, Engine) and faq_engine.source == "factory"

    side_effect: list[str] = []
    assert not side_effect

    async with dg.ascope() as ascope:
        ascope.register_exit_callback(random_callback)
        # register a callback to be called when scope is exited
        assert isinstance(ascope, Resolver)
        # NOTE: scopes are also resolvers, thus can have sub-scope
        conn = await ascope.aresolve(get_conn)
        # generator function will be transformed into context manager and can only be resolved within scope.
        assert isinstance(conn, Connection)
        assert conn.status == "started"
        # context manager is entered when scoped is entered.
        res = await ascope.aresolve(func_dep)
        assert res == 69
        # function dependencies are also supported

    assert conn.status == "closed"
    # context manager is exited when scope is exited.
    assert side_effect[0] == "callbacked"
    # registered callback will aslo be called.

```




## More

For more detailed information, check out [Documentation](https://raceychan.github.io/ididi)

- Tutorial

- Usage of factory

- Visualize the dependency graph

- Circular Dependency Detection

- Error context
