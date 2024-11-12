# DEVNOTE

## Version 0.2.0

- [x] resolve based on static resolve

let dg.resolve depends on static resolve, where the result of static resolve is cached and used when resolve the same dependent

## Version 0.3.0

let DependencyGraph incharge of building nodes, currently node.from_param and _create_sig would build nodes recursively.
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