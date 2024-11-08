# Changelog

## [0.1.1] - 2024-10-29

### Added

- Support for Generic types
- Support for Forward references
- Circular dependencies detection(beta)

## [0.1.2] - 2024-10-30

### Fixed

- Factory return generic types

### Features

- [x] (beta) Visualizer to visualize the dependency graph

### Improvements

- [x] support more resource types
- [x] support more builtin types

## version 0.1.3

- [x] bettr typing support
- [x] node config

## version 0.1.4

- [x] static resolve
- [x] async context manager with graph
- [x] fix: when exit DependencyGraph, iterate over all nodes would cause rror when builtin types are involved
- [x] feat: DependencyGraph.factory would statically resolve the dependent and return a factory

## version 0.1.5

- [x] fix: when a dependency is a union type, the dependency resolver would not be able to identify the dependency

- [x] fix: variadic arguments would cause UnsolvableDependencyError, instead of injection an empty tuple or dict

## version 0.2.0

- [x] feat: lazy dependent

## version 0.2.1

- [x] fix: static resolve at __aenter__ would cause dict changed during iteration error

## version 0.2.2

- [x] feat: adding py.typed file

## version 0.2.3

- [x] fix: change implementation of DependencyGraph.factory to be compatible with fastapi.Depends

- [x] fix: fix a bug with subclass of ty.Protocol

- [x] feat: dg.resolve depends on static resolve

- [x] feat: node.build now support nested, partial override

e.g.

```python
class DataBase:
    def __init__(self, engine: str = "mysql", /, driver: str = "aiomysql"):
        self.engine = engine
        self.driver = driver

class Repository:
    def __init__(self, db: DataBase):
        self.db = db

class UserService:
    def __init__(self, repository: Repository):
        self.repository = repository


node = DependentNode.from_node(UserService)
instance = node.build(repository=dict(db=dict(engine="sqlite")))
assert isinstance(instance, UserService)
assert isinstance(instance.repository, Repository)
assert isinstance(instance.repository.db, DataBase)
assert instance.repository.db.engine == "sqlite"
assert instance.repository.db.driver == "aiomysql"
```

- [x] feat: detect unsolveable dependency with static resolve

## version 0.2.4

Features:

- feat: support for async entry
- feat: adding `solve` as a top level api

Improvements:

- resolve / static resolve now supports factory
- better typing support for DependentNode.resolve
- dag.reset now supports clear resolved nodes
