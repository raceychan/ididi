# ididi

## Introduction

ididi is a zero-configuration, minimal-code-intrusiveness dependency injection library for Python that works out of the box.

It allows you to define dependencies in a declarative way without any boilerplate code.

## Install

```bash
pip install ididi
```

## Usage

### Decorate your top level dependencies and leave the rest to ididi

```python
from didi import DependencyGraph

dg = DependencyGraph()

class Config:
    def __init__(self, env: str = "prod"):
        self.env = env


class Database:
    def __init__(self, config: Config):
        self.config = config


class Cache:
    def __init__(self, config: Config):
        self.config = config


class UserRepository:
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache


class AuthService:
    def __init__(self, db: Database):
        self.db = db

@dg.node
class UserService:
    def __init__(self, repo: UserRepository, auth: AuthService):
        self.repo = repo
        self.auth = auth

@dg.node
def auth_service_factory(database: Database) -> AuthService:
    return AuthService(db=database)

service = dg.resolve(UserService)
assert isinstance(service.repo.db, Database)
assert isinstance(service.repo.cache, Cache)
assert isinstance(service.auth.db, Database)
assert service.auth.db is service.repo.db
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


### Runtime override is also supported

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


## Features

- Automatically resolve dependencies
- Circular dependencies detection
- Support Forward References

### Resolve Rules

- If a node has a factory, it will be used to create the instance.
- Otherwise, the node will be created using the `__init__` method.
  - Parent's `__init__` will be called if no `__init__` is defined in the node.
- bulitin types are not resolvable by nature, it requires default value to be provided.
- runtime override with `dg.resolve`
