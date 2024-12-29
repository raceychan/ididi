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

## version 1.0.7

adding support for python 3.13

## version 1.0.8

improvements:

now ididi will look for conflict in reusability,
For example,
when a dependency with `reuse=False` has a dependent with `reuse=True`, ididi would raise ReusabilityConflictError when statically resolve the nodes.

```bash
ididi.errors.ReusabilityConflictError: Transient dependency `Database` with reuse dependents
make sure each of AuthService -> Repository is configured as `reuse=False`
```

## version 1.0.9

- remove `import typing as ty`, `import typing_extensions as tyex` to reduce global lookup

- add an `ignore` field to node config, where these ignored param names or types will be ignored at statical resolve / resolve phase

```py
@dg.node(ignore=("name", int))
class User:
    name: str
    age: int

dg = DependencyGraph()
dg.node(ignore=("name", int))(User)
with pytest.raises(TypeError):
    dg.resolve(User)

n = dg.resolve(User, name="test", age=3)
assert n.name == "test"
assert n.age == 3
```

- fix a potential bug where when resolve dependent return None or False it could be re-resolved

- rasie error when a factory returns a builtin type

```py
@dg.node
def create_user() -> str:
    ...
```

## version 1.0.10

- fix a bug where error would happen when user use `dg.resolve` to resolve a resource factor, example:

```py
def user_factory() -> ty.Generator[User, None, None]:
    u = User(1, "test")
    yield u
dg.static_resolve(user_factory)
with dg.scope() as scope:
    # this would fail before 1.0.10
    u = scope.resolve(user_factory)
```

improvement
improve typing support for scope.resolve, now it recognize resource better.

## version 1.1.0

Feat:

- add a special mark so that users don't need to specifically mark `dg.node`

```py
async def create_user(user_repo: inject(repo_factory)):
    ...

# so that user does not have to explicitly declear repo_factory as a node

dg = DependencyGraph()

@dg.node
def repo_factory():
    ...
```

- support config to entry

- support DependencyGraph as dependency

```py
def get_user_service(dg: DependencyGraph) -> UserService:
    return dg.resolve(UserService)

dg.resolve(get_user_service)
# this would pass the same graph into function
```

## version 1.1.1

- remove `static_resolve` config from DependencyGraph
- remove `factory`  method frmo DependencyGraph
- add a new `self_inejct` config to DependencyGraph
- add `register_dependent` config to DependencyGraph, SyncScope, AsyncScope
- inject now supports both as annotated annotation and as default value, as well as nested annotated annotation

```py
class APP:
    def __init__(self, graph: DependencyGraph):
        self._graph = graph
        self._graph.register_dependent(self)

dg = DependencygGraph()
app = APP(graph)


@app.register
async def login(app: APP):
    assert app is app
```

## version 1.1.2

if we need a Context object in a non context manner, e.g.

```py
class Service:
    async def __aenter__(self): ...
    async def __aexit__(self, *args): ...
        
def get_service() -> Service:
    return Service()

dg.resolve(get_service)
```

- factory_type
- factory override order

improvements on `entry`

- `entry` now supports positional argument.
- `entry` now would use current scope
- 100% performance boost on `entry`
- separate `INodeConfig` and `IEntryConfig`

## version 1.1.3

Improvements:

- rename `inject` to `use`, to avoid the implication of `inject` as if it was to say only param annotated with `inject` will be injected.
whereas what it really means is which factory to use

- further optimize `entry`, now when the decorated function does not depends on resource,
it will not create a scope.

In 1.1.2

```bash
0.089221 seoncds to call call regular function create_user 100000 times
0.412887 seoncds to call entry version of create_user 100000 times
```

now at 1.1.3

```bash
0.099846 seoncds to call regular function create_user 100000 times
0.104534 seoncds to call entry version of create_user 100000 times
```

This make functions that does not depends on resource 4-5 times faster than 1.1.2
Fix:

- :sparkles: fix a bug where when a dependent with is registered with `DependencyGraph.register_dependnet`, it will still be statically resolved.

## version 1.1.4

Improvements:

- Add `DependencyGraph.should_be_scoped` api to check if a dependent type contains any resource dependency, and thus should be scoped. this is particularly useful when user needs to (dynamically) decide whether they should create the resource in a scope.

Fix:

- previously entry only check if any of its direct dependency is rousource or not, this will cause bug when any of its indirect dependencies is a resource, raise OutOfScope Exception

## version 1.1.5

Fix:

- previously only resource itself will be managed by scope, now if a dependnet depends on a resource, it will also be managed by scope.

## version 1.1.6

Fix:

- fix a bug where if user menually decorate its async generator / sync factory with contextlib.asynccontextmanager / contextmanager, `DependencyNode.factory_type` would generated as `function`.

.e.g:

```py
from typing import AsyncGenerator
from contextlib import asynccontextmanager

@asynccontextmanager
async def get_client() -> AsyncGenerator[Client, None]:
    client = Client()
    try:
        yield client
    finally:
        await client.close()
```

- improvement

- use a dedicate ds to hold singleton dependent,
- improve error message for missing annotation / unresolvable dependency 
