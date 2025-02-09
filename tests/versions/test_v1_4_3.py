from ididi import Graph

"""
- separate overrides and resolved, carry resolved to sub dependencies.
- using threadpool in scoped sync function.

```python
class SyncScope:
    def enter_context(self):
        self._thread_pool.execute(self.statck.enter_context, context)

    def __exit__(self):
        # use thread pool executor
        ...
```

- create a default scope for each graph
- rename `Graph.scope` to `Graph.create_scop`, reserve `Graph.scope` to the dfault scope

- since we create node via `dg.node`, if a dependency in graph._ignore, we ignore those when create node.dependencies, so that we don't have to do this in `node.unsolved_params`

- builtin types with default no longer considered as dependencies
"""


def test_graph_ignore_apply_create_node():
    dg = Graph(ignore="name")

    class User:
        def __init__(self, name: str):
            self.name = name

    dg.node(User)
    assert not dg.nodes[User].dependencies


def test_node_dpes_ignore_builtin_with_default():
    dg = Graph()

    class User:
        def __init__(self, name: str = "name", age: int = 3):
            self.name = name
            self.age = age

    dg.node(User)
    assert not dg.nodes[User].dependencies
