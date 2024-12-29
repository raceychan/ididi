from collections import defaultdict
from types import MappingProxyType
from typing import Any, Callable, Union

from ._type_resolve import get_bases
from .node import DependentNode
from .utils.param_utils import MISSING, Maybe
from .utils.typing_utils import T

GraphNodes = dict[type[T], DependentNode[T]]
"""
### mapping a type to its corresponding node
"""

GraphNodesView = MappingProxyType[type[T], DependentNode[T]]
"""
### a readonly view of GraphNodes
"""

ResolvedInstances = dict[type[T], T]
"""
mapping a type to its resolved instance
"""

TypeMappings = dict[type[T], list[type[T]]]
"""
### mapping a type to its dependencies
"""


class BaseRegistry:
    """Base registry class with common functionali"""

    _mappings: dict[type, Any]
    __slots__ = ("_mappings",)

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, dependent_type: type) -> bool:
        return dependent_type in self._mappings

    def clear(self) -> None:
        self._mappings.clear()


class TypeRegistry(BaseRegistry):
    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: TypeMappings[Any] = defaultdict(list)

    def __getitem__(self, dependent_type: type[T]) -> list[type[T]]:
        return self._mappings[dependent_type].copy()

    def update(self, other: "TypeRegistry"):
        self._mappings.update(other._mappings)

    def register(self, dependent_type: type[T]) -> None:
        self._mappings[dependent_type].append(dependent_type)
        for base in get_bases(dependent_type):
            self._mappings[base].append(dependent_type)

    def remove(self, dependent_type: type):
        for base in get_bases(dependent_type):
            self._mappings[base].remove(dependent_type)

        del self._mappings[dependent_type]

    def get(
        self,
        dependent_type: Union[type[T], Callable[..., T]],
        /,
        default: Maybe[list[type[T]]] = MISSING,
    ) -> Maybe[list[type[T]]]:
        return self._mappings.get(dependent_type, default)


class ResolutionRegistry(BaseRegistry):
    __slots__ = ("_mappings",)

    def __init__(self):
        # should use weakref dict
        self._mappings: ResolvedInstances[Any] = dict()

    def __getitem__(self, dependent_type: Union[type[T], Callable[..., T]]) -> T:
        return self._mappings[dependent_type]

    def remove(self, dependent_type: type) -> None:
        self._mappings.pop(dependent_type, None)

    def update(self, other: "ResolutionRegistry"):
        self._mappings.update(other._mappings)

    def register(
        self, dependent_type: Union[type[T], Callable[..., T]], instance: T
    ) -> None:
        if isinstance(dependent_type, type):
            instance_type: type[T] = type(instance)
            for base in get_bases(instance_type):
                if base in self._mappings:
                    continue
                self._mappings[base] = instance
        self._mappings[dependent_type] = instance

    def get(
        self,
        dependent_type: Union[type[T], Callable[..., T]],
        /,
        default: Maybe[T] = MISSING,
    ) -> Maybe[T]:
        return self._mappings.get(dependent_type, default)


class Visitor:
    __slots__ = ("_nodes",)

    def __init__(self, nodes: GraphNodes[Any]):
        self._nodes = nodes

    def _dfs(
        self,
        start_types: Union[list[type], type],
        pre_visit: Union[Callable[[type], None], None] = None,
        post_visit: Union[Callable[[type], None], None] = None,
    ) -> None:
        """Generic DFS traversal with customizable visit callbacks.

        Args:
            start_types: Starting type(s) for traversal
            pre_visit: Called before visiting node's dependencies
            post_visit: Called after visiting node's dependencies
        """
        if isinstance(start_types, type):
            start_types = [start_types]

        visited = set[type]()

        def _get_deps(node_type: type) -> list[type]:
            node = self._nodes[node_type]
            return [
                p.param_type for _, p in node.signature if p.param_type in self._nodes
            ]

        def dfs(node_type: type):
            if node_type in visited:
                return
            visited.add(node_type)

            if pre_visit:
                pre_visit(node_type)

            for dep_type in _get_deps(node_type):
                dfs(dep_type)

            if post_visit:
                post_visit(node_type)

        for node_type in start_types:
            dfs(node_type)

    def get_dependents(self, dependency: type) -> list[type]:
        dependents: list[type] = []

        def collect_dependent(node_type: type):
            node = self._nodes[node_type]
            if any(p.param_type is dependency for _, p in node.signature):
                dependents.append(node_type)

        self._dfs(list(self._nodes), pre_visit=collect_dependent)
        return dependents

    def get_dependencies(self, dependent: type, recursive: bool = False) -> list[type]:
        if not recursive:
            return [p.param_type for _, p in self._nodes[dependent].signature]

        def collect_dependencies(t: type):
            if t != dependent:
                dependencies.append(t)

        dependencies: list[type] = []
        self._dfs(
            dependent,
            post_visit=collect_dependencies,
        )
        return dependencies

    def top_sorted_dependencies(self) -> list[type]:
        "Sort the whole graph, from lowest dependencies to toppest dependents"
        order: list[type] = []
        self._dfs(list(self._nodes), post_visit=order.append)
        return order
