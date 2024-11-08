# ididi

## Introduction

Ididi is a pythonic dependency injection lib, with ergonomic apis, without boilplate code, works out of the box.

It allows you to define dependencies in a declarative way without any boilerplate code.

## Source Code

[Github-ididi](https://github.com/raceychan/ididi)

## Install

```bash
pip install ididi
```

To view viusal dependency graph, install `graphviz`

```bash
pip install ididi[graphviz]
```

## Usage

### Decorate your top level dependencies and leave the rest to ididi

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

```python
from ididi import entry

class EmailService: ...


class EventStore: ...

@entry
def main(email: EmailService, es: EventStore) -> str:
    assert isinstance(email, EmailService)
    assert isinstance(es, EventStore)
    assert email.user.auth.db.config.env == es.db.config.env
    return "ok"

assert main() == "ok"
```

### Usage with FastAPI

```python
from fastapi import FastAPI
from ididi import DependencyGraph

app = FastAPI()
dg = DependencyGraph()

class AuthService: ...

@dg.node
def auth_service_factory() -> AuthService:
    return AuthService()

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

when a node is defined as 'lazy', each of its dependency will be delayed to be resolved as much as possible.

Note that 'lazy' is transitive, if `ServiceA` is lazy, and `ServiceA` depends on `ServiceB`, then `ServiceB` is also lazy.

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

assert instance.user_repo.test() == "test" # user_repo would be resolved when user_repo.test is accessed.
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
- bulitin types are not resolvable by nature, it requires default value to be provided.
- runtime override with `dg.resolve`
