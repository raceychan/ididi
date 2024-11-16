
## Terminology

> what do we mean by these words

`dependent`: a class, or a function that requires arguments to be built/called.

`dependency`: an object that is required by a dependent.

`resource`: a dependent that implements the contextlib.AbstractAsync/ContextManager, or has an async/sync generator as its factory, is considered a resource.

`static resolve`: resursively build node from dependent, but does not create the instance of the dependent type.

`resolve`: recursively resolve the dependent and its dependencies, then create an instance of the dependent type.

`solve`: an alias for `resolve`

`entry`: a special type of node, where it has no dependents and its factory is itself.

## What and why

### What is Dependency Injection ?

If a class requires other classes as its attributes, then these attributes are regarded as dependencies of the class, and the class requiring them is called a dependent.

```python
class Downloader:
    def __init__(self, session: requests.Session):
        self.session = session
```

Here, `Downloader` is a dependent, with `requests.Session` being its dependency.

Dependency injection means dynamically constructing the instances of these dependency classes and then pass them to the dependent class.

the same class without dependency injection looks like this:

```python
class Downloader:
    def __init__(self):
        self.session = requests.Session(url=configured_url, timeout=configured_timeout)
```

Now, since `requests.Session` is automatically built with `Downloader`, it would be difficult to change the behavior of `requests.Session` at runtime.


### Why do we need it?

There are actually a few reasons why you might not need it, the most fundamental one being your code does not need reuseability and flexibility.

1. If you are writing a script that only runs when you menually execute it, and it is often easier to rewrite the whole script than to modify it,
then it probably more efficient to program everything hard-coded. This is actually a common use case of python,
DEVOPS, DataAnalysts, etc.

For example, you can actually modify the dependencies of a class at runtime.

```python
class Downloader:
    ...

downloader = Downloader()
downloader.session = requests.Session(url=configured_url, timeout=configured_timeout)
```

However, this creates a few problems:

- It is error-prone, you might forget to modify the dependencies, or you might modify the dependencies in the wrong order.
- It is not typesafe, you might pass the wrong type of dependencies to the class.
- It is hard to track when the dependencies are modified.

Dependency injection enables you to extend the dependencies of a class without modifying the class itself, which increases the flexibility and reusability of the class.


### Do we need a DI framework? 

Not necessarily, You will be doing just fine using menual dependency injection,as long as the number of dependencies in your app stays within a managable range.

It gets more and more helpful once your your dependency graph starts getting more complicated,
For example, you might have something like this in your app,

where you menually inject dependencies into dependent.

<details>
    <summary>
    complicated service construction
    </summary>

```python title="factory.py"
from .infra import * 

def auth_service_factory(
    settings: Settings,
) -> AuthService:
    connect_args = (
        settings.db.connect_args.model_dump()
    )
    execution_options = (
        settings.db.execution_options.model_dump()
    )
    engine = engine_factory(
        db_url=settings.db.DB_URL,
        echo=settings.db.ENGINE_ECHO,
        isolation_level=settings.db.ISOLATION_LEVEL,
        pool_pre_ping=True,
        connect_args=connect_args,
        execution_options=execution_options,
    )
    async_engine = sqlalchemy.ext.asyncio.AsyncEngine(engine)
    db = AsyncDatabase(async_engine)
    cache = RedisCache[str].build(
        url=config.URL,
        keyspace=config.keyspaces.APP,
        socket_timeout=config.SOCKET_TIMEOUT,
        decode_responses=config.DECODE_RESPONSES,
        max_connections=config.MAX_CONNECTIONS,
    )
    token_registry = TokenRegistry(cahce=cache, 
        token_bucket=TokenBucket(cache, key=Settings.redis.bucket_key)
    )
    uow = UnitOfWork(db)
    encryptor = Encryptor(
        secret_key=settings.security.SECRET_KEY.get_secret_value(),
        algorithm=settings.security.ALGORITHM,  
    )
    eventstore =  EventStore(uow)
    auth_repo = AuthRepository(db)
    auth_service = AuthService(
        auth_repo=auth_repo,
        token_registry=token_registry,
        encryptor=encryptor,
        eventstore=eventstore,
        security_settings=settings.security,
    )
    return auth_service
```

</details>


But then you realize that some of these dependencies should be shared across your services,
for example, auth repo might be needed by both AuthService and UserService, or even more

You might also need to menually create and manage scope as some resources should be accessed/shared only whtin a certain scope, e.g., a request.


### Why Ididi?

ididi helps you do this while stays out of your way, you do not need to create additional classes like `Container`, `Provider`, `Wire`, nor adding lib-specific annotation like `Closing`, `Injectable`, etc.

ididi provides unique powerful features that most alternatives don't have, such as support to inifinite number of context-specific nested sopce, lazydependent, advanced circular dependency detection, plotting, etc.