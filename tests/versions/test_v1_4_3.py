from ididi import Graph, Ignore, use

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

- *args , **kwargs without UnPack no longer considered as dependencies.
- builtin types with provided default no longer considered as dependencies.
- dependency.unresolvabale is now an attribute, instead of a property.
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


# def test_reuse_resolved():
#     def dependency(a: int) -> Ignore[int]:
#         return a

#     def main(a: int, b: int, c: int = use(dependency)) -> Ignore[float]:
#         return a + b + c

#     dg = Graph()

#     dg.resolve(main)

from typing import Any, NewType


class Request:
    ...


RequestParams = NewType("RequestParams", dict[str, Any])


# async def test_resolve_request():
#     dg = Graph()

#     async def resolve_request(r: Request) -> RequestParams:
#         return RequestParams({"a": 1})

#     dg.node(resolve_request)
#     await dg.resolve(resolve_request, r=Request())
