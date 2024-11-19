# Tutorial

## Two primitives

In most cases, you will only need these two primitive method for buliding your dpendent/dependency

- `dg.resolve`

you can either resolve a dependent class

```py
dg = DependencyGraph()

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

scope is like DependencyGraph, but for resouces.
you can pass them around as you need,

```py
async with dg.scope() as scope:
    conn = await scope.resolve(Connection)
    await exec_sql(conn)
```

### Usage with FastAPI

```python title="app.py"
from fastapi import FastAPI
from ididi import DependencyGraph

app = FastAPI()
dg = DependencyGraph()

def auth_service_factory(db: DataBase) -> AuthService:
    async with dg.scope() as scope
        yield dg.resolve(AuthService)

Service = ty.Annotated[AuthService, Depends(dg.factory(auth_service_factory))]

@app.get("/")
def get_service(service: Service):
    return service
```

Stay tune.

More advanced usage example coming...

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

**`DependencyGraph.node` accepts a wide arrange of types, such as dependent class, sync/async facotry, sync/async resource factory, with typing support.**
