# Features

## Core

### Automatic dependencies injection

You can use generator/async generator to create a resource that needs to be closed.
NOTE:

1. resources, if set to be reused, will be shared across different dependents only within the same scope, and destroyed when the scope is exited.
2. async resource in a sync dependent is not supported, but sync resource in a async dependent is supported.  

```python
from ididi import DependencyGraph

dg = DependencyGraph()

@dg.node
async def get_db(client: Client) -> ty.AsyncGenerator[DataBase, None]:
    db = DataBase(client)
    assert client.is_opened
    try:
        await db.connect()
        yield db
    finally:
        await db.close()

@dg.entry
async def main(db: DataBase, sql: str) -> ty.Any:
    res = await db.execute(sql)
    return res

assert await main(sql="select money from bank")
```

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
    res =  Resource()
    yield res
    res.close()

with dg.scope() as scope:
    resource = scope.resolve(Resource)

# For async generator
async with dg.scope() as scope:
    resource = await scope.resolve(Resource)
```

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
Two scopes or more with the same name would follow most recent rule.

#### Nested Nmaed Scope

```python
async with dg.scope(app_name) as app_scope:
    async with dg.scope(router) as router_scope:
        async with dg.scope(endpoint) as endpoint_scope:
            async with dg.scope(user_id) as user_scope:
                async with dg.scope(request_id) as request_scope:
                    ...
```

For any functions called within the request_scope, you can get the most recent scope with `dg.use_scope()`,
or its parent scopes, i.e. `dg.use_scope(app_name)` to get app_scope.

Note that since scope in context-specific, you will need to pass your scope to new thread if needed.

For example, To use scope in background task, you would need to explicitly pass scope to your task

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

### Visualize the dependency graph

> [!NOTE] You will need to install graphviz to be able to use Visualizer


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

## Beta

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
