<<<<<<< HEAD
=======
# ididi

- [ididi](#ididi)
  - [Introduction](#introduction)
    - [Source Code Link](#source-code-link)
  - [Install](#install)
  - [DOCS](#docs)
  - [Usage](#usage)
    - [Quick Start](#quick-start)
    - [Terminology](#terminology)
    - [Automatic dependencies injection](#automatic-dependencies-injection)
    - [Using Scope to manage resources](#using-scope-to-manage-resources)
    - [Usage with FastAPI](#usage-with-fastapi)
    - [Visualize the dependency graph(beta)](#visualize-the-dependency-graphbeta)
    - [Lazy Dependency(Beta)](#lazy-dependencybeta)
    - [Runtime override](#runtime-override)
    - [Advanced Usage](#advanced-usage)
      - [ABC](#abc)
        - [Register ABC implementation with `dg.node`](#register-abc-implementation-with-dgnode)
        - [Multiple Implementations of ABC](#multiple-implementations-of-abc)
    - [Resolve Rules](#resolve-rules)
    - [What and why](#what-and-why)
      - [What is dependency injection?](#what-is-dependency-injection)
      - [Why do we need it?](#why-do-we-need-it)
    - [FAQ](#faq)
      - [How do I override, or provide a default value for a dependency?](#how-do-i-override-or-provide-a-default-value-for-a-dependency)
      - [how do i override a dependent in test?](#how-do-i-override-a-dependent-in-test)
      - [How do I make ididi reuse a dependencies across different dependent?](#how-do-i-make-ididi-reuse-a-dependencies-across-different-dependent)

>>>>>>> a48f988 (Release version 0.3.1)
## Introduction

ididi is a pythonic dependency injection lib, with ergonomic apis, without boilplate code, works out of the box.

ididi is designed to be non-intrusive, it stays out of your existing code, and is easy to be added/removed.

ididi is 100% strictly typed and well-tested, you can expect excellent typing support.

### Source Code

[ididi-github](https://github.com/raceychan/ididi)

### Docs
[ididi-docs](https://raceychan.github.io/ididi/)

## Install

```bash
pip install ididi
```

To view viusal dependency graph, install `graphviz`

```bash
pip install ididi[graphviz]
```

## DOCS

https://raceychan.github.io/ididi/

## Usage

### Quick Start

```python
import ididi

class Config:
    def __init__(self, env: str = "prod"):
        self.env = env

class Database:
    def __init__(self, config: Config):
        self.config = config

class UserRepository:
    def __init__(self, db: Database):
        self.db = db

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

assert isinstance(ididi.solve(UserService), UserService)
```

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
        db.open()
        yield db
    finally:
        await db.close()


@dg.entry
async def main(db: DataBase) -> str:
    assert db.is_opened
    return "ok"

assert await main() == "ok"
```

### Using Scope to manage resources

you might use combination of `with` or `async with` statement and `dg.scope()` to manage resources.

resources will only be shared across dependents only withint the same scope,
and will be automatically destryoed and closed when the scope is exited.

NOTE: classes that implment `contextlib.AbstractContextManager` or `contextlib.AbstractAsyncContextManager` are also considered to be resources and can be resolved within scope.

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

### Usage with FastAPI

NOTE: resource is supported

```python
from fastapi import FastAPI
from ididi import DependencyGraph

app = FastAPI()
dg = DependencyGraph()

class AuthService: ...

@dg.node
def auth_service_factory(db: DataBase) -> AuthService:
    asycn with AuthService(db=db) as auth:
        yield auth

Service = ty.Annotated[AuthService, Depends(dg.factory(auth_service_factory))]

@app.get("/")
def get_service(service: Service):
    return service
```

### Visualize the dependency graph(beta)

```python
from ididi import DependencyGraph, Visualizer
dg = DependencyGraph()
vs = Visualizer(dg)

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
vs.view # use vs.view in jupyter notebook, or use vs.save(path, format) otherwise
```

![image](https://github.com/user-attachments/assets/b86be121-3957-43f3-b75c-3689a855d7fb)

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

### Advanced Usage

#### ABC

##### Register ABC implementation with `dg.node`

you should use `dg.node` to let ididi know about the implementations of the ABC.
you are going to resolve.

```python
from abc import ABC, abstractmethod
class Repository(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the repository data."""
        pass

@dag.node
class Repo1(Repository):
    def save(self) -> None:
        pass

@dag.node
class Repo2(Repository):
    def save(self) -> None:
        pass

dag.resolve(Repository)
```

You might also use `__init_subclass__` hook to automatically register implementations.

##### Multiple Implementations of ABC

ididi will use the last implementation registered to resolve the ABC, you can use a factory to override this behavior.

```python
class Repository(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the repository data."""
        pass

@dag.node
class Repo1(Repository):
    def save(self) -> None:
        pass

@dag.node
class Repo2(Repository):
    def save(self) -> None:
        pass

@dag.node
def repo_factory() -> Repository:
    return Repo1()

assert Repository in dag.nodes

repo = dag.resolve(Repository)
assert isinstance(repo, Repo1)
```

### Resolve Rules

- If a node has a factory, it will be used to create the instance.
- Otherwise, the node will be created using the `__init__` method.
  - Parent's `__init__` will be called if no `__init__` is defined in the node.
- whenver there is a default value, it will be used to resolve the dependency.
- bulitin types are not resolvable by nature, it requires default value to be provided.
- runtime override with `dg.resolve`

### What and why

#### What is dependency injection?

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

#### Why do we need it?

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

### Terminology

`dependent`: a class, or a function that requires arguments to be built/called.

`dependency`: an object that is required by a dependent.

`resource`: a dependent that implements the contextlib.AbstractAsync/ContextManager, or has an async/sync generator as its factory, is considered a resource.

`static resolve`: resursively build node from dependent, but does not create the instance of the dependent type.

`resolve`: recursively resolve the dependent and its dependencies, then create an instance of the dependent type.

`solve`: an alias for `resolve`

`entry`: a special type of node, where it has no dependents and its factory is itself.

### FAQ

#### How do I override, or provide a default value for a dependency?

you can use `dg.node` to create a factory to override the value.
you can also have dependencies in your factory, and they will be resolved recursively.

```python
class Config:
    def __init__(self, env: str = "prod"):
        self.env = env

@dg.node
def config_factory() -> Config:
    return Config(env="test")
```

#### how do i override a dependent in test?

you can use `dg.node` with a factory method to override the dependent resolution.

```python
class Cache: ...

class RedisCache(Cache):
    ...

class MemoryCache(Cache):
    ...

@dg.node
def cache_factory(...) -> Cache:
    return RedisCache() 

in your conftest.py:

@dg.node
def memory_cache_factory(...) -> Cache:
    return MemoryCache()    
```

as this follows LSP, it works both with ididi and type checker.

#### How do I make ididi reuse a dependencies across different dependent?

by default, ididi will reuse the dependencies across different dependent,
you can change this behavior by setting `reuse=False` in `dg.node`.

```python
@dg.node(reuse=False) # True by default
class AuthService: ...
```

