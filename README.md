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

## Usage

### Quick Start

```python
from ididi import DependencyGraph, use, entry, AsyncResource

async def conn_factory(engine: AsyncEngine) -> AsyncGenerator[AsyncConnection, None]:
    async with engine.begin() as conn:
        yield conn

class UnitOfWork:
    def __init__(self, conn: AsyncConnection=use(conn_factory)):
        self._conn = conn

@entry
async def main(command: CreateUser, uow: UnitOfWork):
    await uow.execute(build_query(command))
```

**No Container, No Provider, No Wiring, just *Python***

## Features

### Automatic dependencies injection

You can use generator/async generator to create a resource that needs to be closed.
NOTE:

1. resources, if set to be reused, will be shared across different dependents only within the same scope, and destroyed when the scope is exited.
2. async resource in a sync dependent is not supported, but sync resource in a async dependent is supported.  

```python
from ididi import use, entry, DependencyGraph

async def get_db(dg: DependencyGraph, client: Client, repository: Repository) -> ty.AsyncGenerator[DataBase, None]:
    db = DataBase(repository, client)
    try:
        await db.connect()
        yield db
    finally:
        await db.close()

@entry
async def main(sql: str, db: DataBase = use(get_db)) -> ty.Any:
    res = await db.execute(sql)
    return res

assert await main("select money from bank")
```

> [!NOTE]
> **`DependencyGraph.node` accepts a wide arrange of types, such as dependent class, sync/async facotry, sync/async resource factory, with typing support.**

### Using Scope to manage resources

- **Infinite nested scope is supported.**
- **Parent scope can be accssed by child scope(within the same context)**
- **Resources will be shared across dependents only withint the same scope(reuse needs to be True)**
- **Resources will be automatically closed and destroyed when the scope is exited.**
- **Classes that implment `contextlib.AbstractContextManager` or `contextlib.AbstractAsyncContextManager` are also considered to be resources and can/should be resolved within scope.**

- **Scopes are separated by context**

> [!NOTE]
If you have two call stack of `a1 -> b1` and `a2 -> b2`,
    Here `a1` and `a2` are two calls to the same function `a`,
    then, in `b1`, you can only access scope created by the `a1`, not `a2`.

This is particularly useful when you try to separate resources by route, endpoint, request, etc.

#### Async, or not, works either way

```python
@dg.node
def get_resource() -> ty.Generator[Resource, None, None]:
    res = Resource()
    yield res
    res.close()

with dg.scope() as scope:
    resource = scope.resolve(Resource)

# For async generator
async with dg.scope() as scope:
    resource = await scope.resolve(Resource)
```

> [!NOTE]
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

### Tutorial

There are cases where you would like to menually build the dependency yourself with a factory function,

#### Usage with FastAPI

```python
from fastapi import FastAPI
from ididi import DependencyGraph

app = FastAPI()
dg = DependencyGraph()

def auth_service_factory() -> AuthService:
    async with dg.scope() as scope
        yield scope.resolve(AuthService)

Service = ty.Annotated[AuthService, Depends((auth_service_factory))]

@app.get("/")
def get_service(service: Service):
    return service
```

>[!IMPORTANT]
DependencyGraph does NOT have to be a global singleton

Although we use `dg` extensively to represent an instance of DependencyGraph for the convenience of explaination,
it **DOES NOT** mean it has to be a *global singleton*. These are some examples you might inject it into your fastapi app at different levels.

(ididi v1.0.5)
You can create different dependency graphs in different files, then merge them in the main graph

```py
from app.features.user import user_graph
from app.features.auth import auth_graph
from app.features.payment import payment_graph

# in your entry file

dg = DependencyGraph()
dg.merge([user_graph, auth_graph, payment_graph])
```

##### DependencyGraph as an app-level instance

```py
import typing as ty

from fastapi.routing import APIRoute, APIRouter
from starlette.types import ASGIApp, Receive, Scope, Send

from ididi import DependencyGraph


class GraphedScope(ty.TypedDict):
    dg: DependencyGraph


@asynccontextmanager
async def lifespan(app: FastAPI | None = None) -> ty.AsyncIterator[GraphedScope]:
    dg = DependencyGraph()
    async with dg.scope("app") as dg:
        yield {"dg": dg}


@app.post("/users")
async def signup_user(request: Request):
    dg = request.state.dg
    service = dg.resolve(UserService)
    user_id = await service.signup_user(...)
    return user_id

```

##### Injecting DependencyGraph at route level

```py
class UserRoute(APIRoute):
    def get_route_handler(self) -> ty.Callable[[Request], ty.Awaitable[Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            dg = DependencyGraph()
            request.scope["dg"] = dg

            async with dg.scope() as user_scope:
                response = await original_route_handler(request)
                return response

        return custom_route_handler

user_router = APIRouter(route_class=UserRoute)
```

##### Injecting DependencyGraph at request level

```py
class GraphedMiddleware:
    def __init__(self, app, dg: DependencyGraph):
        self.app = app
        self.dg = dg

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        # NOTE: remove follow three lines would break lifespan
        # as startlette would pass lifespan event here
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope["dg"] = self.dg
        await self.app(scope, receive, send)


app.add_middleware(GraphedMiddleware, dg=DependencyGraph)
```

##### with background task

To use scope in background task, you would need to explicitly pass scope to your task

```py
@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, dg.use_scope(), email, message="some notification")
    return {"message": "Notification sent in the background"}

def write_notification(scope: SyncScope, email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)
        scope.resolve(MessageQueue).publish("Email Sent")

    # To search parent scope:
    parent_scope = scope.get_scope(name)
```

### Resolve Order

`DependencyGraph.resolve`, rsolves dpendent with this order:

1. override
2. default value
3. factory

### Usage of factory

If you are working with non-trivial web application, there is a good chance that you already
have some factory functions used to build your dependnecies, to get them work with ididi, just decorate them with `dg.node`.

- Menually config details of external libraries

```python
@dg.node
def engine_factory(config: Config)-> sqlalchemy.engine.Engine:
    engine = create_engine(
        url=config.db.URL,
        pool_recycle=config.db.POOL_RECYCLE,
        isolation_level=config.db.ISOLATION_LEVEL
    )
    return engine
```

- Privide a stub for your dependencies for testing.

```py
@dg.node
def fake_engine_factory(config: Config)-> sqlalchemy.engine.Engine:
    return FakeEngine()

@pytest.fixture
def engine():
    return dg.resolve(Engine)
```

- Provide different implementation of the dependencies based on some condition.

```py
@dg.node
def redis_cache(config: Config) -> redis.Redis:
    if config.RUNTIME_ENV == 'prod':
        return redis.Redis(...)
    return redis.Redis(...)
```

- Assign a implementation for parent class

```py
class Storage:
    ...
class Database(Storage):
    ...
class S3(Storage): 
    ...

@dg.node
def storage_factory(config: Config) -> Storage:
    if config.storage.storage_type = "cold":
        return S3(...)
    return Database(...)
```

> This works for ABC, typing.Protocol, as well as plain classes.

### Visualize the dependency graph(beta)

```python
from ididi import DependencyGraph, Visualizer
dg = DependencyGraph()

class ConfigService:
    def __init__(self, env: str = "test"):
        self.env = env


class DatabaseService:
    def __init__(self, config: ConfigService):
        self.config = config


class CacheService:
    def __init__(self, config: ConfigService):
        self.config = config


class BaseService:
    def __init__(self, db: DatabaseService):
        self.db = db


class AuthService(BaseService):
    def __init__(self, db: DatabaseService, cache: CacheService):
        super().__init__(db)
        self.cache = cache


class UserService:
    def __init__(self, auth: AuthService, db: DatabaseService):
        self.auth = auth
        self.db = db


class NotificationService:
    def __init__(self, config: ConfigService):
        self.config = config


class EmailService:
    def __init__(self, notification: NotificationService, user: UserService):
        self.notification = notification
        self.user = user

dg.static_resolve(EmailService)
vs = Visualizer(dg)
vs.view # use vs.view in jupyter notebook, or use vs.save(path, format) otherwise
vs.save(path, format)
```

![image](https://github.com/user-attachments/assets/b86be121-3957-43f3-b75c-3689a855d7fb)

### Circular Dependency Detection

ididi would detect if circular dependency exists, if so, ididi would give you the circular path

For example:

```python
class A:
    def __init__(self, b: "B"):
        self.b = b


class B:
    def __init__(self, a: "C"):
        self.a = a


class C:
    def __init__(self, d: "D"):
        pass


class D:
    def __init__(self, a: A):
        self.a = a


def test_advanced_cycle_detection():
    """
    DependentNode.resolve_forward_dependency
    """
    dag = DependencyGraph()

    with pytest.raises(CircularDependencyDetectedError) as exc_info:
        dag.static_resolve(A)
    assert exc_info.value.cycle_path == [A, B, C, D, A]
```

You can call `DependencyGraph.static_resolve_all` on app start to statically resolve all
your noded classes, and let ididi get ready for resolve them at upcoming calls.

### Lazy Dependency(Beta)

you can  use `@dg.node(lazy=True)` to define a dependent as `lazy`,
which means each of its dependency will not be resolved untill accessed.

start with v0.3.0, lazy node is no longer transitive.

```python
class UserRepo:
    def __init__(self, db: Database):
        self._db = db

    def test(self):
        return "test"

@dg.node(lazy=True)
class ServiceA:
    def __init__(self, user_repo: UserRepo, session_repo: SessionRepo):
        self._user_repo = user_repo
        self._session_repo = session_repo

        assert isinstance(self._user_repo, LazyDependent)
        assert isinstance(self._session_repo, LazyDependent)

    @property
    def user_repo(self) -> UserRepo:
        return self._user_repo

    @property
    def session_repo(self) -> SessionRepo:
        return self._session_repo

assert isinstance(instance.user_repo, LazyDependent)
assert isinstance(instance.session_repo, LazyDependent)

# user_repo would be resolved when user_repo.test() is called.
assert instance.user_repo.test() == "test" 
```

### Runtime override

```python
dg = DependencyGraph()

class Inner:
    def __init__(self, value: str = "inner"):
        self.value = value

@dg.node
class Outer:
    def __init__(self, inner: Inner):
        self.inner = inner

# Override nested dependency
instance = dg.resolve(Outer, inner=Inner(value="overridden"))
assert instance.inner.value == "overridden"
```

### Error context

static resolve might fail when class contain unresolvable dependencies, when failed, ididi would show a chain of errors like this:

```bash
ididi.errors.MissingAnnotationError: Unable to resolve dependency for parameter: env in <class 'Config'>,
annotation for `env` must be provided

<- Config(env: _Missed)
<- DataBase(config: Config)
<- AuthService(db: DataBase)
<- UserService(auth: AuthService)
```

Where UserService depends on AuthService, which depends on Database, then Config, but Config.env is missing annotation.

To solve error like this, you can either provide the missing annotation, or you can provide a factory method with menually provide the value.

```python
@dg.node
def config_factory() -> Config:
    return Config(env=".env")
```

this is more common when you want to menually config external libraries like redis, sqlalchemy, etc.

```python
@dg.node
def redis_factory(config: Config) -> redis.Redis:
    return Redis(url=config.redis_url)
```
