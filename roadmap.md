# DEVNOTE

## Version 0.2.0

- [x] resolve based on static resolve

let dg.resolve depends on static resolve, where the result of static resolve is cached and used when resolve the same dependent

## Version 0.3.0

let DependencyGraph incharge of building nodes, currently node.from_param and `_create_sig` would build nodes recursively.
since they do not have access to the parent graph, they have no way to know whether a node is already registered in the graph and should not be built again.

this is make ididi fail to work in some cases, for example:

```python
from pathlib import Path

class Config:
    def __init__(self, env: pathlib.Path) -> None:
        self.env = env

@dg.node
def config_factory() -> Config:
    return Config(env=Path.cwd())

class Cache:
    def __init__(self, config: Config) -> None:
        self.config = config

@dg.node
class Registry:
    def __init__(self, cache: Cache) -> None:
        self.cache = cache
```

Here Config depends on env, where it can't not be built, but user provide a factory for env.
But since dg.node would build node recursively, it would build env anyway, which cause error.

In cases where a dependency is unsolvable by nature, and requires a factory to be provided, this would cause the node creation to fail, even if user did provide the factory.
Since there is no way to tell if the unsolvable dependency is being provided with a factory or not, given the toppest level node does not know about the graph during its creation.

In the near future, dg.node will only create a node without recursively resolving its sub dependencies,
it only resolve the dependent and the dependencies of the dependent, nothing more.

Node would be fully resolved when `dg.analyze` is called.

## version 1.0.0

use contextvar.contextvar for global scope

## version 1.1.0

1. threading
2. reuse should be reversed-transitive, meaning that a reused dependent that depends on a non-reused dependency should become non-reused as well