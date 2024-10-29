# ididi

## Introduction

ididi is a dependency injection library for Python. It allows you to define dependencies in a declarative way and resolve them at runtime.

ididi is under active, heavy development. The API is not stable and will change.
But it will remains simple and minimalistic.

## Install

```bash
pip install ididi==0.1.0
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

@dag.node
class UserService:
    def __init__(self, repo: UserRepository, auth: AuthService):
        self.repo = repo
        self.auth = auth

@dag.node
def auth_service_factory(database: Database) -> AuthService:
    return AuthService(db=database)

service = dag.resolve(UserService)
assert isinstance(service.repo.db, Database)
assert isinstance(service.repo.cache, Cache)
assert isinstance(service.auth.db, Database)
assert service.auth.db is service.repo.db
```

### Runtime override is also supported

```python
dag = DependencyGraph()

class Inner:
    def __init__(self, value: str = "inner"):
        self.value = value

@dag.node
class Outer:
    def __init__(self, inner: Inner):
        self.inner = inner

# Override nested dependency
instance = dag.resolve(Outer, inner=Inner(value="overridden"))
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
- bulitin types are not resolvable by nature, and it requires default value to be provided.
- runtime override with `dag.resolve`
