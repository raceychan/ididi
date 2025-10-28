# import cython
from types import FunctionType, MappingProxyType
from typing import Any, Callable, Hashable, Union, cast

from ._node import DependentNode
from ._type_resolve import get_bases
from .interfaces import IDependent
from .utils.typing_utils import T

GraphNodes = dict[IDependent[Any], DependentNode]
"""
### mapping a type to its corresponding node
"""

GraphNodesView = MappingProxyType[IDependent[Any], DependentNode]
"""
### a readonly view of GraphNodes
"""

ResolvedSingletons = dict[Any, Any]
"""
mapping a type to its resolved instance, only instances of reusable node will be added here.
"""

TypeMappings = dict[IDependent[T], list[IDependent[T]]]
"""
### mapping a type to its dependencies
"""


class TypeRegistry(dict[Hashable, list[IDependent[Any]]]):
    def register(self, dependent: IDependent[T]) -> None:
        try:
            self[dependent].append(dependent)
        except KeyError:
            self[dependent] = [dependent]

        for base in get_bases(dependent):
            try:
                self[base].append(dependent)
            except KeyError:
                self[base] = [dependent]

    def remove(self, dependent_type: IDependent[T]) -> None:
        for base in get_bases(dependent_type):
            self[base].remove(dependent_type)
        del self[dependent_type]


class Visitor:
    __slots__ = ("_nodes",)

    def __init__(self, nodes: GraphNodes):
        self._nodes = nodes

    def _visit(
        self,
        start_types: Union[list[IDependent[Any]], IDependent[Any]],
        pre_visit: Union[Callable[[IDependent[Any]], None], None] = None,
        post_visit: Union[Callable[[IDependent[Any]], None], None] = None,
    ) -> None:
        """Generic DFS traversal with customizable visit callbacks.

        Args:
            start_types: Starting type(s) for traversal
            pre_visit: Called before visiting node's dependencies
            post_visit: Called after visiting node's dependencies
        """
        if isinstance(start_types, type) or isinstance(start_types, FunctionType):
            start_types = [start_types]

        visited = set[IDependent[Any]]()

        def _get_deps(node_type: IDependent[Any]) -> list[IDependent[Any]]:
            node = self._nodes[node_type]
            return [
                p.param_type
                for p in node.dependencies
                if p.param_type in self._nodes
            ]

        def dfs(node_type: IDependent[Any]):
            if node_type in visited:
                return
            visited.add(node_type)

            if pre_visit:
                pre_visit(node_type)

            for dep_type in _get_deps(node_type):
                dfs(dep_type)

            if post_visit:
                post_visit(node_type)

        for node_type in cast(list[IDependent[Any]], start_types):
            dfs(node_type)

    def get_dependents(self, dependency: IDependent[Any]) -> list[IDependent[Any]]:
        dependents: list[IDependent[Any]] = []

        def collect_dependent(node_type: IDependent[Any]):
            node = self._nodes[node_type]
            if any(p.param_type is dependency for p in node.dependencies):
                dependents.append(node_type)

        self._visit(list(self._nodes), pre_visit=collect_dependent)
        return dependents

    def get_dependencies(
        self, dependent: IDependent[Any], recursive: bool = False
    ) -> list[IDependent[Any]]:
        if not recursive:
            return [p.param_type for p in self._nodes[dependent].dependencies]

        def collect_dependencies(t: IDependent[Any]):
            if t != dependent:
                dependencies.append(t)

        dependencies: list[IDependent[Any]] = []
        self._visit(
            dependent,
            post_visit=collect_dependencies,
        )
        return dependencies

    def top_sorted_dependencies(self) -> list[IDependent[Any]]:
        "Sort the whole graph, from lowest dependencies to toppest dependents"
        order: list[IDependent[Any]] = []
        self._visit(list(self._nodes), post_visit=order.append)
        return order
