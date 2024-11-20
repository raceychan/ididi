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

## version 0.2.5

Improvements:

- [x] better error message for node creation error
NOTE: this would leads to most exception be just NodeCreationError, and the real root cause would be wrapped in NodeCreationError.error.

This is because we build the DependencyNode recursively, and if we only raise the root cause exception without putting it in a error chain, it would be hard to debug and notice the root cause.

e.g.

```bash
.pixi/envs/test/lib/python3.12/site-packages/ididi/graph.py:416: in node
    raise NodeCreationError(factory_or_class, "", e, form_message=True) from e
E   ididi.errors.NodeCreationError: token_bucket_factory()
E   -> TokenBucketFactory(aiocache: Cache)
E   -> RedisCache(redis: Redis)
E   -> Redis(connection_pool: ConnectionPool)
E   -> ConnectionPool(connection_class: idid.Missing)
E   -> MissingAnnotationError: Unable to resolve dependency for parameter: args in <class 'type'>, annotation for `args` must be provided
```

Features:

## version 0.2.6

Feature:

- you can now use async/sync generator to create a resource that needs to be closed

```python
from ididi import DependencyGraph
dg = DependencyGraph()

@dg.node
async def get_client() -> ty.AsyncGenerator[Client, None]:
    client = Client()
    try:
        yield client
    finally:
        await client.close()


@dg.node
async def get_db(client: Client) -> ty.AsyncGenerator[DataBase, None]:
    db = DataBase(client)
    try:
        yield db
    finally:
        await db.close()


@dg.entry_node
async def main(db: DataBase):
    assert not db.is_closed
```

## version 0.2.7

Improvements:

- support sync resource in async dependent
- better error message for async resource in sync function
- resource shared within the same scope, destroyed when scope is exited

## version 0.2.8

FIX:

Adding a temporary fix to missing annotation error, which would be removed in the future.

Example:

```python
class Config:
    def __init__(self, a):
        self.a = a

@dag.node
def config_factory() -> Config:
    return Config(a=1)

class Service:
    def __init__(self, config: Config):
        self.config = config

dg.resolve(Service)
```

This would previously raise `NodeCreationError`, with root cause being `MissingAnnotationErro`, now being fixed.

## version 1.0.0

API changes:
`ididi.solve` renamed to `ididi.resolve` for better consistency with naming style.
`ididi.entry` no longer accepts `INodeConfig` as kwargs. 

Feat:

DependencyGraph.use_scope to indirectly retrive nearest local
raise `OutOfScopeError`, when not within any scope.

Fix:

- now correctly recoganize if a class is async closable or not
- resolve factory with forward ref as return type.

Improvements:

- raise PositionalOverrideError when User use `dg.resolve` with any positional arguments.

- test coverages raised from 95% to 99%, where the only misses only exists in visual.py, and  are import error when graphviz not installed, and graph persist logic.

- user can directly call `Visualizer.save` without first calling `Visualizer.make_graph`.


## version 1.0.1

Improvements

- performance boost on dg.static_resolve

## version 1.0.2

Features

- named scope, user can now call `dg.use_scope(name)` to get a specific parent scope in current context, this is particularly useful in scenario where you have tree-structured scopes, like app-route-request.


## version 1.0.3

Fix:

- now config of `entry` node is default to reuse=False, which means the result will no longer be cached.
- better typing with `entry`


## version 1.0.4

improvements:
refactor visitor


## version 1.0.5

improvements:
DependencyGraph now supports `__contains__` operator, `a in dg` is the equivalent to `a in dg.nodes`

features:
DependencyGraph now supports a `merge` operator, example:

```py
from app.features.users import user_dg

dg = DependencyGraph()
dg.merge(user_dg)
```

you might choose to merge one or more DependencyGraph into the main graph, so that you don't have to import a single dg to register all your classes.