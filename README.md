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

```python
from didi import DependencyGraph

dg = DependencyGraph()

@dag.node
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
```
