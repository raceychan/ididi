# Performance

ididi is very efficient and performant, with average time complexity of O(n)

#### `Graph.statical_resolve` (type analysis on each class, can be done at import time)

Time Complexity: O(n) - O(n**2)

O(n): average case, where each dependent has a constant number of dependencies, for example, each dependents has on average 3 dependencies.

O(n**2): worst case, where each dependent has as much as possible number of dependencies, for example, with 100 nodes, node 1 has 99 dependencies, node 2 has 98, etc.

I personally don't think anyone would ever encouter the worse case in real-world, but even if someone does, you can still expect ididi resolve thousand of such classes in seoncds.

As a reference:

tests/test_benchmark.py 0.003801 seoncds to statically resolve 122 classes

#### `Graph.resolve` (inject dependencies and build the dependent instance)

Time Complexity: O(n)

You might run the benchmark yourself with following steps

1. clone the repo, cd to project root
2. install pixi from [pixi](https://pixi.sh/latest/)
3. run `pixi install`
4. run `make benchmark`

#### Performance tip

- use dg.node to decorate your classes

- use dg.node to decorate factory of third party classes so that ididi does not need to analyze them

For Example

```python
def redis_factory(settings: Settings) -> Redis:
    # build redis here
    return redis
```

- use dg.analyze_nodes when your app starts, which will statically resolve all your classes decorated with @dg.node.


However, ididi is so performant so most of time there is no reason to apply these tips.

### Resolve Rules

- If a node has a factory, it will be used to create the instance.
- Otherwise, the node will be created using the `__init__` method.
  - Parent's `__init__` will be called if no `__init__` is defined in the node.
- whenver there is a default value, it will be used to resolve the dependency.
- bulitin types are not resolvable by nature, it requires default value to be provided.
- runtime override with `dg.resolve`
