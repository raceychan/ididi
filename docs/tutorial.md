# Tutorial

## Two primitives

In most cases, you will only need these two primitive method for buliding your dpendent/dependency

- `dg.resolve`

you can either resolve a dependent class

```py
dg = Graph()

dg.resolve(UserService)
```

or a factory function that return the dependent class.

```py
def get_user_service(user_repo: UserRepo) -> UserService:
    return UserService(user_repo)

dg.resolve(get_user_service)
```

or both

```py
def get_user_service() -> UserService:
    return dg.resolve(UserService)
```

- `dg.scope`

scope is like Graph, but for resouces.
you can pass them around as you need,

```py
async with dg.ascope() as scope:
    conn = await scope.resolve(Connection)
    await exec_sql(conn)
```

## Usage with FastAPI

```python title="app.py"
from fastapi import FastAPI
from ididi import Graph

app = FastAPI()
dg = Graph()

def auth_service_factory() -> AuthService:
    async with dg.ascope() as scope
        yield dg.resolve(AuthService)

Service = ty.Annotated[AuthService, Depends(auth_service_factory)]

@app.get("/")
def get_service(service: Service):
    return service
```

>[!NOTE] Graph does NOT have to be a global singleton

Although we use `dg` extensively to represent an instance of Graph for the convenience of explaination,
it **DOES NOT** mean it has to be a *global singleton*. These are some examples you might inject it into your fastapi app at different levels.

### Graph as an app-level instance

```py
import typing as ty

from fastapi.routing import APIRoute, APIRouter
from starlette.types import ASGIApp, Receive, Scope, Send

from ididi import Graph


class GraphedScope(ty.TypedDict):
    dg: Graph


@asynccontextmanager
async def lifespan(app: FastAPI | None = None) -> ty.AsyncIterator[GraphedScope]:
    async with Graph() as dg:
        yield {"dg": dg}


@app.post("/users")
async def signup_user(request: Request):
    dg = request.state.dg
    service = dg.resolve(UserService)
    user_id = await service.signup_user(...)
    return user_id

```

#### Injecting Graph at route level

```py
class UserRoute(APIRoute):
    def get_route_handler(self) -> ty.Callable[[Request], ty.Awaitable[Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            dg = Graph()
            request.scope["dg"] = dg

            async with dg.ascope() as user_scope:
                response = await original_route_handler(request)
                return response

        return custom_route_handler

user_router = APIRouter(route_class=UserRoute)
```

#### Injecting Graph at request level

```py
class GraphedMiddleware:
    def __init__(self, app, dg: Graph):
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


app.add_middleware(GraphedMiddleware, dg=Graph)
```


## Usage of factory

### Using factory to override dependency injection

There are cases where you would like to menually build the dependency yourself with a factory function,

#### Menually config details of external libraries

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

**`Graph.node` accepts a wide arrange of types, such as dependent class, sync/async facotry, sync/async resource factory, with typing support.**
