# Changelog

## version 1.8.2

### Highlights

- `NodeMeta` and `use` are now public exports, so applications can declare factories or metadata without reaching into private modules.
- `resolve_meta` now preserves `ignore` flags and infers factories from `Annotated[Service, use()]`, keeping ignored dependencies and default factories consistent during analysis and scope checks.

### Details

- `ididi.__init__` re-exports `NodeMeta` and `use` alongside other public helpers.
- `Graph` analysis and `should_be_scoped` respect `NodeMeta(ignore=True)` in nested `Annotated` metadata, so ignored parameters stay skipped and `UnsolvableNodeError.__notes__` still include dependent/parameter context when resolution fails.
- `use()` markers without explicit factories now default to the annotated type, matching earlier behavior while keeping the new metadata path intact.

## version 1.8.1

### Highlights

- `Ignore[...]` is now implemented via `NodeMeta(ignore=True)`, so the legacy `IGNORE_PARAM_MARK`
  sentinel is gone. Any `typing.Annotated` blocks that previously used the constant can switch to
  `ididi.Ignore` (or embed their own `NodeMeta(ignore=True)` marker) with no behavioral change.
- `NodeMeta` gained the new `ignore` flag, and all `use(...)` helpers / graph analysis logic now honor
  it when deciding whether a parameter should participate in dependency resolution.

### Details

- Removed `IGNORE_PARAM_MARK` from the public config surface and runtime checks.
- Updated `Ignore` helper to expand to `Annotated[T, NodeMeta(ignore=True)]`.
- `resolve_type_from_meta`, `Dependency.should_ignore`, and `_from_function` now look for the new
  metadata instead of literal sentinels, so nested `Annotated` stacks keep functioning.
- Tests and documentation were refreshed to showcase the new marker form.

## version 1.7.7

### Highlights
- `Dependency.annotation` now stores the exact annotation object (including `typing.Annotated` metadata), so analysis tooling can interrogate user-defined markers without recomputing them.
- `Dependency` and `DependentNode` are now slot-based data classes—no `__dict__`—which shrinks each node and makes accidental attribute writes fail fast.

### Details
- **Annotation preservation** (`tests/versions/test_v1_7_7.py:20`-`tests/versions/test_v1_7_7.py:30`): running `graph.analyze(get_user)` keeps the original `Annotated[str, meta(...)]` object on `node.dependencies[0].annotation`, letting advanced callers recover the metadata that produced the dependency.
- **Slot-only nodes** (`tests/versions/test_v1_7_7.py:31`-`tests/versions/test_v1_7_7.py:35`, `ididi/_node.py`): both `Dependency` and `DependentNode` now declare `__slots__`, so reaching for `__dict__` raises `AttributeError` and per-node memory drops because Python no longer allocates dictionaries for them.

```python
from dataclasses import dataclass
from typing import Annotated
import pytest

from ididi import Graph, Ignore

@dataclass
class Meta:
    name: str
    version: str

def meta(name: str, version: str) -> Meta:
    return Meta(name=name, version=version)

def get_user(query: Annotated[str, meta("query_param", "v1")]) -> Ignore[str]:
    return f"User({query})"

graph = Graph()
node = graph.analyze(get_user)
dependency = node.dependencies[0]

assert dependency.annotation == Annotated[str, meta("query_param", "v1")]
with pytest.raises(AttributeError):
    dependency.__dict__
with pytest.raises(AttributeError):
    node.__dict__
```

The preserved `annotation` makes it trivial to read custom metadata back out of analyzed dependencies, while `__slots__` keeps heavily-used graph structures lean and predictable in large applications.

## version 1.7.6

### Highlights
- Node reuse now starts as "unspecified", giving decorator registration and `use()` hints a chance to agree before conflicts are raised.
- Parameter-level reuse clashes now surface as `ParamReusabilityConflictError`, pointing directly at the factory parameter that introduced the mismatch.
- `use()` return annotations propagate reuse flags to their nodes, so factory overrides can mark themselves reusable without extra bookkeeping.

### Details
- **Missing-as-default reuse** (`tests/versions/test_v1_7_6.py:22`): analyzing a dependency that uses `use(get_user, reuse=False)` flips the previously unspecified node to non-reusable, and a conflicting `reuse=True` hint now fails fast with `ParamReusabilityConflictError`.
- **Return annotations drive node config** (`tests/versions/test_v1_7_6.py:35`): factories that return `Annotated[T, use(reuse=True)]` mark the target reusable immediately; later attempts to re-register them with `reuse=False` raise `ConfigConflictError`.
- **Explicit overrides stay authoritative** (`tests/versions/test_v1_7_6.py:50`, `tests/versions/test_v1_7_6.py:66`): decorator-specified reuse continues to win, and any subsequent override that disagrees is rejected with `ConfigConflictError`.
- **Public `is_provided` helper** (`ididi/__init__.py:34`): expose `is_provided` so callers can tell whether a node's reuse flag has been set before deciding on overrides.

### Behavior Change

Nodes now keep their reuse flag unset until code explicitly opts in, because `_reuse` starts as the sentinel `MISSING` during node construction (`ididi/_node.py:349`-`ididi/_node.py:392`). When the same dependent is included again we run through `DependentNode.update_reusability`; any side that still has `_reuse` equal to `MISSING` simply adopts the provided value, so `ConfigConflictError` only fires when both registrations made an explicit, conflicting choice (`ididi/graph.py:700`-`ididi/graph.py:708`). `use(...)` markers participate in the same flow: they publish their reuse preference while analysis walks the parameter graph, and we only escalate to `ParamReusabilityConflictError` when an earlier declaration disagrees (`ididi/graph.py:551`-`ididi/graph.py:563`, `tests/versions/test_v1_7_6.py:22`-`tests/versions/test_v1_7_6.py:47`). The public `is_provided` helper lets applications check whether a node already committed to a reuse value before attempting overrides (`ididi/__init__.py:34`, `tests/versions/test_v1_7_6.py:26`, `tests/versions/test_v1_7_6.py:72`).

```python
from typing import Annotated
import pytest

from ididi import Graph, use, is_provided
from ididi.errors import ConfigConflictError, ParamReusabilityConflictError

graph = Graph()

class User: ...

# First registration – leaves reuse as the sentinel MISSING
graph.node(User)
assert not is_provided(graph.nodes[User].reuse)

# A handler pins the dependency to reuse=False; this now succeeds
class LoginHandler:
    def __init__(self, user: Annotated[User, use(reuse=False)]):
        self.user = user

graph.analyze(LoginHandler)
assert graph.nodes[User].reuse is False  # value captured only now

# A conflicting handler tries to demand reuse=True; we now get a targeted error
class ProfileHandler:
    def __init__(self, user: Annotated[User, use(reuse=True)]):
        self.user = user

with pytest.raises(ParamReusabilityConflictError):
    graph.analyze(ProfileHandler)

# Direct overrides behave the same way: once reuse is explicit, any opposite
# declaration fails fast with ConfigConflictError.
with pytest.raises(ConfigConflictError):
    graph.node(User, reuse=True)

# If you really need an opposite policy, supply it before anyone else fixes reuse:
graph2 = Graph()
graph2.node(User, reuse=True)          # explicit upfront
class AuditHandler:
    def __init__(self, user: User): ...
graph2.analyze(AuditHandler)           # fine, reuse already decided
```

Previously `graph.node(User)` defaulted to a boolean reuse flag, so any later `use(reuse=True)`/`use(reuse=False)` annotation collided instantly. The sentinel-driven workflow leaves the decision open until a caller makes an explicit choice, keeping decorator registration, annotations, and overrides in sync and ensuring conflicts only occur when two intentional policies disagree.

## version 1.7.4

### Highlights
- `Graph.node` now accepts `reuse` directly and honours `use(..., reuse=True)` hints end-to-end, so reusable services stay cached even when the dependency is registered via a decorator.
- Reuse configuration conflicts now raise `ConfigConflictError` during analysis, resolution, and graph merges instead of silently overriding the previous node.
- `Graph.entry` keeps the reuse semantics expressed in annotations, letting entry handlers participate in the same caching rules as the rest of the graph.

### Details
- **Decorator-specified reuse** (`tests/test_graph.py::test_dependent_conflicts`):
  ```python
  @g.node(reuse=True)
  class Service: ...

  @g.node
  def handler(dep: Annotated[Service, use(reuse=True)]) -> Handler:
      return Handler(dep)

  g.analyze_nodes()
  assert g.resolve(handler).dep is g.resolve(handler).dep
  ```
  The `reuse=True` flag survives both the class registration and the annotated dependency, ensuring downstream factories receive the cached singleton.

- **Fail-fast reuse conflicts** (`tests/test_graph.py::test_graph_analyze_reuse_dependentcy`, `tests/test_graph_additional.py::test_merge_nodes_reuse_conflict`):
  ```python
  @dg.node
  def user_factory() -> Annotated[User, use(reuse=True)]:
      return User()

  @dg.node
  def manager(user: Annotated[User, use(user_factory, reuse=False)]) -> Manager:
      return Manager(user)

  with pytest.raises(ConfigConflictError):
      dg.resolve(manager)
  ```
  Conflicting `reuse` settings on the same dependency now abort analysis/merge instead of picking an arbitrary winner.

- **Entry wrappers respect reuse hints** (`tests/features/test_entry.py::test_entry_with_default_use`):
  ```python
  async def main(service: Annotated[Service, use(reuse=True)]) -> None:
      ...

  wrapped = graph.entry(main)
  await wrapped()
  ```
  Decorating an entry no longer forces `reuse=False`; annotations decide whether the dependency is cached.

- **Async scopes keep singleton callbacks** (`tests/test_graph_additional.py::test_async_scope_registers_reuse_singleton`):
  ```python
  scope = graph._create_ascope("async")
  await scope.__aenter__()

  instance = Service()
  await scope.aresolve_callback(instance, Service, "function", True)

  assert scope._resolved_singletons[Service] is instance
  ```
  Async scope resolution now records reusable instances the same way the synchronous scope already did.

## version 1.7.3
- support using `Annotated[T, use(..., factory)]` syntax in node
- Added a deprecation warning when `Graph.add_nodes` receives `(node, config)` tuples; this form will be removed in 1.8.0 and callers should migrate to `Graph.add_nodes(use(...))`.
  ```python
  from ididi import Graph, use

  class Service: ...

  graph = Graph()

  # Deprecated style 
  graph.add_nodes((Service, {"reuse": True, "ignore": [1, 2, 3]}))

  # Preferred replacement
  graph.add_nodes(use(Service, reuse=True))
  ```

## version 1.7.2

### Fix:

a quick fix, now raise node config conflicts error for case like this:

```python
@pytest.mark.debug
def test_graph_analyze_reuse_dependentcy():
    dg = Graph()

    class User: ...

    @dg.node
    def user_factory() -> Annotated[User, use(reuse=True, ignore=(5))]:
        return User()

    class UserManager:
        def __init__(self, user: User):
            self.user = user


    @dg.node
    def user_manager_factory(user: Annotated[User, use(user_factory, reuse=False, ignore=(1,2,3))]) -> UserManager:
        return UserManager(user)


    with pytest.raises(ConfigConflictError):
        dg.analyze(user_manager_factory)

```

## version 1.7.1

### Highlights
- Graph.merge now raises `ConfigConflictError` when two graphs offer the same dependency with conflicting node config.
- `use()` can be called with configuration overrides only; the annotated type becomes the default target.
- `use` markers now work in return annotations, so factories can tune node config directly from their type hints.

### Details
- **ConfigConflictError on merge**: merging graphs that both provide a dependency at the same resolve priority now fails fast if the configs disagree. The regression test below (`tests/test_graph.py::test_merge_rejects_conflicting_configs`) captures the behaviour:
  ```python
  import pytest
  from ididi import Graph
  from ididi.errors import ConfigConflictError

  def test_merge_rejects_conflicting_configs():
      g1 = Graph()
      g2 = Graph()

      class Cache: ...

      g1.node(Cache, reuse=True)
      g2.node(Cache, reuse=False)

      with pytest.raises(ConfigConflictError):
          g1.merge(g2)
  ```
- **Default `use` target**: when only config keywords are supplied (`use(reuse=True)`), the annotated dependency is used automatically. See `tests/features/test_entry.py::test_entry_with_default_use`:
  ```python
  from typing import Annotated

  from ididi import Graph, use

  async def test_entry_with_default_use():
      class A: ...

      async def main(a: Annotated[A, use(reuse=True)]) -> None:
          assert isinstance(a, A)

      graph = Graph()
      await graph.entry(main)()
  ```
- **Return-annotation support**: factories can embed `use()` markers in their return type to config resolvability of its output:

```python
from typing import Annotated
from ididi import Graph, use

def test_graph_analyze_reuse_dependent():
    dg = Graph()

    class User: ...

    def user_factory() -> Annotated[User, use(reuse=True)]:
        return User()

    assert dg.resolve(user_factory) is dg.resolve(user_factory)
```

## version 1.7.0

!!!BREAKING CHANGES!!!

1. change `Graph.entry` dependency declarition to opt-out

before 1.7.0:

```python
from ididi import Graph, Ignore
dg = Graph()

@dg.entry
async def create_user(
    user_name: Ignore[str],
    user_email: Ignore[str],
    service: UserService,
) -> UserService:
    return service
```

before 1.7.0, every param is treated as depenedency, unless declared with "idid.Ignore"

now:

```python
from ididi import Graph, use
dg = Graph()

@dg.entry
async def create_user(
    user_name: str,
    user_email: str,
    service: Annotated[UserService, use(UserService)],
) -> UserService:
    return service
```

only params declared with `Annotated[T, use(Callable[..., T])]` is considered dependency.

2. `NodeConfig.reuse` default to `False`

It turns out that, after months of usage in production, I realized people almost always set reuse=False
the reason being that, in most application you would need to resolve scoped object, and since 

1. scoped object often should not be reused(such as `sqlalchemy.AsyncConnection`), and 
2. once an object is transient, all of its parents have to be transient as well.

## version 1.6.4

### improvements:

deperecate default style `use` 

```python
from ididi import use
class Dep: ...

def get_dep() -> Dep: ...

def get_repo(dep: Dep = use(get_dep)): ...
```

this style of factory is no longer recommended, and will be removed in 1.7.0

use 
```python
from ididi import use
from typing import Annotated

class Dep: ...

def get_dep() -> Dep: ...

def get_repo(dep: Annotated[Dep, use(get_dep)]): ...
```

instead

## version 1.6.3

### Fixes:
- [x] fix a bug where node.config.ignore won't work in should_be_scoped

## version 1.6.2

### Fixes:

- [x] fix a bug where dependent is not a class, or is a complex type would raise TypeError
- [x] fix a bug where "Ignore" does not work for complex union types

## version 1.6.1

- Fixes

fix a bug where `Graph.should_be_scoped` won't ignore Ignored params

```python
def test_ignore_error():
    class Engine:
        def __init__(self, name: Ignore[str]):
            self.name = name


    dg = Graph()

    assert not dg.should_be_scoped(Engine) # this would raise error before 1.6.1
```

## version 1.6.0

- fix 

fix a bug where `dg.node(func_dep, **config)` would ignore config, as we would check if dep is in type registry during analyze time, and func dep is not registered before, so that it would be re-registered without config.

## version 1.5.3

- `Factory method`, 
```python

class InfraBuilder:
    
    @dg.factory
    def repo_maker(self) -> UserRepo:
        return UserRepo

as well as

dg.factory(InfraBuilder().repo_maker)
```

## version 1.5.2

- ~~Make both `Graph` and `Scope` self-injectable, when resolve `Resolver` within a graph, return the graph, if resolve it within a scope, return the scope~~

Cancel, this would scope circular reference itself, 
and user can use `dg.use_scope` to get the current scope, this is not worthy it.

- remove `UnsolvableDependencyError` error, which means that we no longer try to analyze builtin types

## version 1.5.1

- support extra params to dependencies and overrides-reuse, this allows us to do complex dependency resolution, 

```python
dg = Graph()

def dep3(e: int) -> Ignore[str]:
    return "ok"

def dependency(a: int, a2: Annotated[str, use(dep3)]) -> Ignore[int]:
    return a

def dep2(k: str, g: Annotated[int, use(dependency)]) -> Ignore[str]:
    assert g == 1
    return k

def main(
    a: int,
    b: int,
    c: Annotated[int, use(dependency)],
    d: Annotated[str, use(dep2)],
) -> Ignore[float]:
    assert d == "f"
    return a + b + c

assert dg.resolve(main, a=1, b=2, k="f", e="e") == 4
```

the downside is that now `Graph.resolve` is 1.3x slower, we might find a way to reduce this in next few patches, but this feature is really needy right now.


- `Graph.entry` no longer ignore builtin-types by default, now user should specifically ignore builtin params. This is mainly for
1. resolve and entry to have similar behaviors 
2. user might be confused by what will be automatically ignored, Ignore[T] is more explicit

- rollback a behavior introduced in `version 1.4.3`, where Ignored params were not considered as dependencies, now they are dependencies again.

- `Graph.resolve` no longer maintain ParamSpec, since for a factory `Callable[P, T]`, `Graph.resolve` can accept either more params than P, less params P, or no params P at all, it does not make much sense to maintain paramspec anymore

## version 1.5.0

- consider switch to anyio[cancel]
- rewrite ididi in cython
- make ididi resolve overhead < 100%. make it as fast as hard-coded 
factories as possible. 

- support return dependency[cancel]

```python
def json_encode(r: Any) -> bytes:
    return orjson.dumps(r)

type Json[T] = Annotated[T, use(json_encode)]

async def create_user(user_id: str = "u") -> Json[User]:
    return User(user_id=user_id)

assert isinstance(dg.resolve(create_user), bytes)

```

here we will serialize the result of `create_user`
we can also refactor how we deal with resource right now
by applying `scope.enter_context` to result, so instead of 

```python
async def get_conn() -> AsyncGenerator[Connection, None, None]
    ...
```

user can write

```pythpn

def enter_ctx(scope: Scope):
    await scope.resolve()


Scoped = Annotated[T, scope_factory]
```



```python
from ididi import Scoped



class Connection:
    ...

async def aget_conn() -> Scoped[Connection]:
    conn = Connection()
    yield conn


def get_conn() -> Scoped[Connection]:
    conn = Connection()
    yield conn
```

## versio 1.4.5(plan)

- [x] support resolve from class method

```python
class Book:
    @classmethod
    def from_article(cls, a: str) -> "Book":
        return Book()


@pytest.mark.debug
def test_resolve_classmethod():
    dg = Graph()
    b = dg.resolve(Book.from_article, a="5")
    assert isinstance(b, Book)
```

- [x] make `Mark` public, user can use it with `typing.Annotated`

You can rely on the bundled `Ignore[...]` helper:

```python
from ididi import Ignore

def get_conn(url: Ignore[str]):
    ...
```

```python
from ididi import use

def get_repo(conn: Annotated[Connection, use(get_conn)]):
    ...
```

- [x] provide a more non-intrusive way to node config, where user only need to add new code, no modifying current code, not even decorator.

```python
class AuthRepo: ...
class TokenRegistry: ...

def get_auth() -> AuthRepo: ...
def get_registry() -> TokenRegistry: ...

class AuthService:
    def __init__(self, auth_repo: AuthRepo, token: TokenRegistry): ...

dg = Graph()
dg.add_nodes(
    get_conn,
    (get_auth, {"reuse": False, "ignore":"name"})
)
```

- make a re-use params version of `Graph.resolve`, 

```python
from typing import Any, NewType

class Request: ...

RequestParams = NewType("RequestParams", dict[str, Any])


async def test_resolve_request():
    dg = Graph()

    async def resolve_request(r: Request) -> RequestParams:
        return RequestParams({"a": 1})

    dg.node(resolve_request)
    await dg.shared_resolve(resolve_request, r=Request())
```

## version 1.4.4

- refactor scope, now user can create a new scope using `scope.scope`

```python
def test_scope_graph_share_methods():
    dg = Graph()

    with dg.scope() as s1:
        with s1.scope() as s2:
            ...
```

- `Graph.get` / `Scope.get`


*create scope from graph* vs *crate scope from scope*

the created scope will `inherit` resolved instances and registered singletons from its creator, thus

- when create from graph, scope inherit resolved instances and registered singletons from graph

- when create from scope, scope inherit resolved instances and registered singletons from the parent scope.

Example
```python
    dg = Graph()

    class User: ...
    class Cache: ...

    dg_u = dg.resolve(User)

    with dg.scope() as s1:
        s1_u = s1.resolve(User)
        assert s1_u is dg_u
        s1_cache = s1.resolve(Cache)
        dg_cache = dg.resolve(Cache)
        assert s1_cache is not dg_cache

        with s1.scope() as s12:
            assert s12.resolve(User) is dg_u
            s12_cache = s12.resolve(Cache)
            assert s12_cache is s1_cache
            assert s12_cache is not dg_cache

    with dg.scope() as s2:
        s2_u = s2.resolve(User)
        assert s2_u is dg_u
        s2_cache = s2.resolve(Cache)
        assert s2_cache is not s1_cache
```

## version 1.4.3

Improvements:

50% performance boost for `Graph.resolve/aresolve/entry`

Features:

- `Graph` now receives `workers: concurrent.futures.ThreadPoolExecutor` in constructor

- send entering and exiting of `contextmanager` in a diffrent thread when using `AsyncScope`, to avoid blocking.

```python
def get_session() -> Generator[Session, None, None]:
    session = Session()
    with session.begin():
        yield session

async with dg.ascope() as scope:
    ss = await scope.resolve(get_session)
```
Here, entering and exiting `get_session` will be executed in a different thread.

- `Graph` now create a default scope, `Graph.use_scope` should always returns a scope.
- Split `Graph.scope` to `Graph.scope` and `Graph.ascope`.
- Deprecate `DependencyGraph`, `DependencyGraph.static_resolve`, `DependencyGraph.static_resolve_all`
- variadic arguments, such as `*args` or `**kwargs` without `typing.UnPack`, are no longer considered as dependencies.
- Builtin types with provided default are no longer considered as dependencies.
- `Dependency.unresolvabale` is now an attribute, instead of a property.

## version 1.4.2

Function as Dependency

Declear a function as dependency via annotating its return type with `Ignore`.

```python
@dataclass
class User:
    name: str
    role: str

def get_user(config: Config) -> Ignore[User]:
    assert isinstance(config, Config)
    return User("user", "admin")


def validate_admin(
    user: Annotated[User, get_user], service: UserService
) -> Ignore[str]:
    assert user.role == "admin"
    assert isinstance(service, UserService)
    return "ok"

assert dg.resolve(validate_admin) == "ok"
```

Note that since `get_user` returns `Ignore[User]` instead of `User`, it won't be used as factory to resolve `User`.

## version 1.4.1

we should take a smarter approach with `ignore`, for node configured with ignore,
we don't put them in node's dependencies. 
This would be more efficient and make more sense.

```python
def test_ignore_dependences():
    dg = DependencyGraph()

    class User:
        def __init__(self, name: Ignore[str]): ...

    dg.node(User)

    assert len(dg.nodes[User].dependencies) == 0
```

## version 1.4.0

This minor focus on a small refactor on `Scope`, we like the idea that `Scope` is a temporary view of its parent `Graph`, so following change is made:

- both resource and non-resource instances created in scope will stay in the scope
- when a graph create a scope, it shares a copy of its resolved singletons and registered singletons, scope can read them, but can not modify them.

This gives a better separation between `Graph` and `Scope`.

## version 1.3.6

- Fix: `Graph.entry` no longer uses existing scope, instead, always create a new scope

## version 1.3.5

- a quick bug fix, where in 1.3.4, registered_singleton is not shared between graph and scope.

The general rule is that scope can access registered singletons and resolved instances but not vice versa.

## version 1.3.4

- refactor `Scope` and `Graph` to avoid circular reference,

## version 1.3.3

- `Graph.search_node`, search node by name, O(n) complexity

This is mainly for debugging purpose, sometimes you don't have access to the dependent type, but do know the name of it. example

```python
class User: ...

dg = Graph()
dg.node(User)

assert dg.search_node("User").dependent_type is User
```

This is particularly useful for type defined by NewType

```python
UserId = NewType("UserId", str)
assert dg.search_node("UserId")
```


- `Graph.override` 

a helper function to override dependent within the graph

```python
def override(self, old_dep: INode[P, T], new_dep: INode[P, T]) -> None:
```

```python
    dg = DependencyGraph()

    @dg.entry
    async def create_user(
        user_name: str, user_email: str, service: UserService
    ) -> UserService:
        return service

    @dg.node
    def user_factory() -> UserService:
        return UserService("1", 2)

    class FakeUserService(UserService): ...

    dg.override(UserService, FakeUserService)

    service_res = await create_user("1", "2")
    assert isinstance(service_res, FakeUserService)
```

Note that, if you only want to override dependency for `create_user`
you can still just use `create_user.replace(UserService, FakeUserService)`,
and such override won't affect others.

## version 1.3.2

- Now when override entry dependencies  with `entryfunc.replace`, ididi will automatically analyze the override dependency

- `NodeConfig` is now immutable and hashable
- rename `Graph._analyze_entry` to `Graph.analyze_params`,

## version 1.3.1

### Features

Resolve NewType


```py
from uuid import uuid4

UUID = NewType("UUID", str)


def uuid_factory() -> UUID:
    return str(uuid4())

def user_factory(user_id: UUID) -> User:
    return User(user_id=user_id)
```

Why not TypeAlias as well?

1. It does not make sense, since TypeAlias is literally an alias

A = str 

means A is the same as str, and it make no sense to resolve a str.


2. Resolving TypeAlias is not possible in python 3.9

```py
UUID = str

def uuid_factory() -> UUID:
    return uuid4()

when we try to resolve uuid_factory, the return is a type `str`,
not a `TypeAlias`.
```

## version 1.3.0

### performance boost

**400%** performance improvements on `Graph.resolve`, by apply cache whever possible,

In previous benchmark, 
0.000558 seoncds to resolve 100 instances

ididi v1.3.0
0.000139 seoncds to resolve 100 instances


### renaming

- rename `DependencyGraph` to `Graph`
- rename `DependencyGraph.static_resolve` to `Graph.analyze`
- rename `DependencyGraph.static_resolve_all` to `Graph.analyze_nodes`

Original names are kept as alias to new names to avoid breaking changes


### typing support

- support Unpack
```py
class DataBase: ...
class Cache: ...
class Config: ...

class KWARGS(TypedDict):
    db: DataBase
    cache: Cache

class Extra(KWARGS):
    config: Config

class Repo:
    def __init__(self, **kwargs: Unpack[KWARGS]) -> None:
        self.db = kwargs["db"]
        self.cache = kwargs["cache"]

class SubRepo(Repo):
    def __init__(self, **kwargs: Unpack[Extra]):
        super().__init__(db=kwargs["db"], cache=kwargs["cache"])
        self.config = kwargs["config"]

def test_resolve_unpack():
    dg = Graph()
    repo = dg.resolve(Repo)
    assert isinstance(repo.db, DataBase)
    assert isinstance(repo.cache, Cache)
    subrepo = dg.resolve(SubRepo)
    assert isinstance(subrepo.config, Config)
```

- support Literal

```py
def test_resolve_literal():
    dg = Graph()

    class User:
        def __init__(self, name: Literal["user"] = "user"):
            self.name = name

    u = dg.resolve(User)
    assert u.name == "user"
```


### Features

```py
Graph.remove_singleton(self, dependent_type: type) -> None:
"Remove the registered singleton from current graph, return if not found"

Graph.remove_dependent(self, dependent_type: type) -> None
"Remove the dependent from current graph, return if not found"
```

## version 1.2.6

- override class dependencies, entry dependencies

```py
class Repo:
    db: DataBase = use(db_factory)


def fake_db_factory()->DataBase:
    return FakeDB()

dg.node(fake_db_factory)
```

### override entry

```py
def test_entry_replace():
    dg = DependencyGraph()

    def create_user(
        user_name: str, user_email: str, service: UserService
    ) -> UserService:
        return service

    class FakeUserService(UserService): ...

    create_user = dg.entry(reuse=False)(create_user)
    create_user.replace(UserService, FakeUserService)

    res = create_user("user", "user@email.com")
    assert isinstance(res, FakeUserService)


    class EvenFaker(UserService):
        ...

    create_user.replace(service=EvenFaker)
    create_user.replace(UserService, service=EvenFaker)

    res = create_user("user", "user@email.com")
    assert isinstance(res, EvenFaker)
```

## version 1.2.5

- remove `LazyDependent`, since `DependencyGraph` can inject itself to any dependencies it resolve, 
`LazyDependent` can be easily achieved by injecting `DependencyGraph` and lazy resolve

```py
from ididi import ignore

class UserRepo:
    def __init__(self, graph: DependencyGraph, db: Ignore[DataBase]=None):
        self._graph = graph
        self._db = db

    @property
    def db(self):
        return self._graph.resolve(DataBase)
```


- performance boost

25% performance increase to `DependencyGraph.resolve` compare to 1.2.4

Current implementation of dg.resolve is 17.356008 times slower than a hard-coded dependency construction.
given how much extra work ididi does resolving dependency, 
our ultimate goal would be make dg.resolve < 10 times slower than a hard-coded solution.

## version 1.2.4


Improvements:

- minor performance gain, 60% faster for instance resolve, should resolve 0.2 million instances in 1 second, although still 40 times slower than hard-coded factory, we tried replace recursive appraoch with iterative, no luck, only tiny improvements, so we stay at recursive appraoch

Fix:
in previous patch

```py
def service_factory(
    *, db: Database = use(db_fact), auth: AuthenticationService, name: str
) -> UserService:
    return UserService(db=db, auth=auth)


def test_resolve():
    dg = DependencyGraph()

    dg.resolve(service_factory, name="aloha")
```

woud raise `UnsolvableDependencyError`, because `name` is a str without default values, 
when when we resolve it, we will first static_resolve it, which would raise error,
now if dependencies that are provided with overrides won't be statically resolved.

## version 1.2.3

- change `_itypes` to `interfaces`, make it public, since it is unlikely to cause breaking change

- remove `partial_resolve` from DependencyGraph

`partial_resolve` was introduced in `1.2.0`.  it is not a good design,
it is like a global `try except`, which hides problems instead of solving them.


- `Ignore` Annotation


```py
from ididi import Ignore

class User:
    def __init__(self, name: Ignore[str]):
        self.name = name
```

which is equivalent to

```
DependencyGraph().node(ignore="name")(User)
```


both `use` and `Ignore` are designed to be user friendly alternative to `DependencyGraph.node`

## version 1.2.2

- `NodeConfig.ignore` & `GraphConfig.ignore` now supports `TypeAliasType`

only affect type checker

- remove `EntryConfig`, now `entry` share the same config as `NodeConfig`,
the only difference is now `dg.entry(reuse=True)` is possible, whereas it used to be always False, which does not really make sense, user can share the same dependencies across different calls to the entry function.


- `dg._reigster_node`, `dg._remove_node`, are now private, `dg.replace_node` is removed.
these three methods require user provide a `DependencyNode`, but we want to avoid user having to interact directly with `DependencyNode` to lower complexity.

## version 1.2.1

a quick bug fix

```py
from ididi import DependencyGraph


class Time:
    def __init__(self, name: str):
        self.name = name


def test_registered_singleton():
    """
    previously
    dg.should_be_scoped(dep_type) would analyze dependencies of the
    dep_type, and if it contains resolvable dependencies exception would be raised

    since registered_singleton is managed by user,
    dg.should_be_scoped(registered_singleton) should always be False.
    """
    timer = Time("1")

    dg = DependencyGraph()

    dg.register_singleton(timer)

    assert dg.should_be_scoped(Time) is False

```

## version 1.2.0

improvements

- change `Scope.register_dependent` to `Scope.register_singleton` to match the change made in 1.1.7 on `DependencyGraph`. 

- positional ignore for node

```py
def test_ignore():
    dg = DependencyGraph()

    class Item:
        def __init__(self, a: int, b: str, c: int):
            self.a = a
            self.b = b
            self.c = c

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(Item)

    dg.node(ignore=(0, str, "c"))
    with pytest.raises(TypeError):
        dg.resolve(Item)

>>> TypeError: __init__() missing 3 required positional arguments: 'a', 'b', and 'c'
```

- partial resolve

previously, if a dependent requires a unresolvable dependency without default value, it would raise `UnsolvableDependencyError`,
this might be a problem when user wants to resolve the dependent by providing the unresolvable dependency with `dg.resolve`.

now `DependencyGraph` accepts a `partial_resolve` parameter, which would allow user to resolve a dependent with missing unsolvable-dependency.

```py
def test_graph_partial_resolve_with_dep():
    dg = DependencyGraph(partial_resolve=True)

    class Database: ...

    DATABASE = Database()

    def db_factory() -> Database:
        return DATABASE

    class Data:
        def __init__(self, name: str, db: Database = use(db_factory)):
            self.name = name
            self.db = db

    dg.node(Data)

    dg.analyze(Data)

    with pytest.raises(UnsolvableDependencyError):
        data = dg.resolve(Data)

    data = dg.resolve(Data, name="data")
    assert data.name == "data"
    assert data.db is DATABASE
```

## version 1.1.7

improvements

- factory have higher priority than default value

now, for a non-builtin type, ididi will try to resolve it even with default value, 
this is because for classses that needs dependency injection, it does not make sense to have a default value in most cases, and in cases where it does, it most likely be like this

```py
class UserRepos:
    def __init__(self, db: Optional[Database] = None):
        self.db = db
```

and it make more sense to inject `Database` than to ignore it.


- global ignore with `DependencyGraph(ignore=(type))`

```py
class Timer:
    def __init__(self, d: datetime):
        self._d = d

dg = DependencyGraph(ignore=datetime)
with pytest.raises(TypeError):
    dg.resolve(Timer)
```

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

## version 1.1.5

Fix:

- previously only resource itself will be managed by scope, now if a dependnet depends on a resource, it will also be managed by scope.

## version 1.1.4

Improvements:

- Add `DependencyGraph.should_be_scoped` api to check if a dependent type contains any resource dependency, and thus should be scoped. this is particularly useful when user needs to (dynamically) decide whether they should create the resource in a scope.

Fix:

- previously entry only check if any of its direct dependency is rousource or not, this will cause bug when any of its indirect dependencies is a resource, raise OutOfScope Exception

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

## version 1.0.10

- fix a bug where error would happen when user use `dg.resolve` to resolve a resource factor, example:

```py
def user_factory() -> ty.Generator[User, None, None]:
    u = User(1, "test")
    yield u
dg.analyze(user_factory)
with dg.scope() as scope:
    # this would fail before 1.0.10
    u = scope.resolve(user_factory)
```

improvement
improve typing support for scope.resolve, now it recognize resource better.

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

## version 1.0.8

improvements:

now ididi will look for conflict in reusability,
For example,
when a dependency with `reuse=False` has a dependent with `reuse=True`, ididi would raise ReusabilityConflictError when statically resolve the nodes.

```bash
ididi.errors.ReusabilityConflictError: Transient dependency `Database` with reuse dependents
make sure each of AuthService -> Repository is configured as `reuse=False`
```

## version 1.0.7

adding support for python 3.13

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

## version 1.0.4

improvements:
refactor visitor

## version 1.0.3

Fix:

- now config of `entry` node is default to reuse=False, which means the result will no longer be cached.
- better typing with `entry`

## version 1.0.2

Features

- named scope, user can now call `dg.use_scope(name)` to get a specific parent scope in current context, this is particularly useful in scenario where you have tree-structured scopes, like app-route-request.

## version 1.0.1

Improvements

- performance boost on dg.analyze

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

## version 0.2.7

Improvements:

- support sync resource in async dependent
- better error message for async resource in sync function
- resource shared within the same scope, destroyed when scope is exited

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

## version 0.2.4

Features:

- feat: support for async entry
- feat: adding `solve` as a top level api

Improvements:

- resolve / static resolve now supports factory
- better typing support for DependentNode.resolve
- dag.reset now supports clear resolved nodes

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

## version 0.2.2

- [x] feat: adding py.typed file

## version 0.2.1

- [x] fix: static resolve at __aenter__ would cause dict changed during iteration error

## version 0.2.0

- [x] feat: lazy dependent

## version 0.1.5

- [x] fix: when a dependency is a union type, the dependency resolver would not be able to identify the dependency

- [x] fix: variadic arguments would cause UnsolvableDependencyError, instead of injection an empty tuple or dict

## version 0.1.4

- [x] static resolve
- [x] async context manager with graph
- [x] fix: when exit DependencyGraph, iterate over all nodes would cause rror when builtin types are involved
- [x] feat: DependencyGraph.factory would statically resolve the dependent and return a factory

## version 0.1.3

- [x] bettr typing support
- [x] node config

## [0.1.2] - 2024-10-30

### Fixed

- Factory return generic types

### Features

- [x] (beta) Visualizer to visualize the dependency graph

### Improvements

- [x] support more resource types
- [x] support more builtin types

## [0.1.1] - 2024-10-29

### Added

- Support for Generic types
- Support for Forward references
- Circular dependencies detection(beta)
