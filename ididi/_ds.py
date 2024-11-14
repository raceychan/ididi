import typing as ty
from collections import defaultdict
from types import MappingProxyType

from ._type_resolve import get_bases, is_async_closable, is_closable
from .node import DependentNode
from .utils.param_utils import MISSING, Maybe

type GraphNodes[I] = dict[type[I], DependentNode[I]]
"""
### mapping a type to its corresponding node
"""

type GraphNodesView[I] = MappingProxyType[type[I], DependentNode[I]]
"""
### a readonly view of GraphNodes
"""

type ResolvedInstances[T] = dict[type[T], T]
"""
mapping a type to its resolved instance
"""

type TypeMappings[T] = dict[type[T], list[type[T]]]
"""
### mapping a type to its dependencies
"""


class BaseRegistry:
    """Base registry class with common functionality."""

    _mappings: dict[type, ty.Any]

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, dependent_type: type) -> bool:
        return dependent_type in self._mappings

    def clear(self) -> None:
        self._mappings.clear()


class TypeRegistry(BaseRegistry):
    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: TypeMappings[ty.Any] = defaultdict(list)

    def __getitem__[T](self, dependent_type: type[T]) -> list[type[T]]:
        return self._mappings[dependent_type].copy()

    def register[T](self, dependent_type: type[T]) -> None:
        self._mappings[dependent_type].append(dependent_type)
        for base in get_bases(dependent_type):
            self._mappings[base].append(dependent_type)

    def remove(self, dependent_type: type):
        for base in get_bases(dependent_type):
            self._mappings[base].remove(dependent_type)

        del self._mappings[dependent_type]

    def get[
        T
    ](
        self,
        dependent_type: type[T] | ty.Callable[..., T],
        /,
        default: Maybe[list[type[T]]] = MISSING,
    ) -> Maybe[list[type[T]]]:
        return self._mappings.get(dependent_type, default)


class ResolutionRegistry(BaseRegistry):
    __slots__ = ("_mappings",)

    # TODO: close_callback_registry
    # type close_callback[T] = ty.Callable[[T], None | ty.Awaitable[None]]

    def __init__(self):
        self._mappings: ResolvedInstances[ty.Any] = {}

    def __getitem__[T](self, dependent_type: type[T] | ty.Callable[..., T]) -> T:
        return self._mappings[dependent_type]

    def remove(self, dependent_type: type) -> None:
        self._mappings.pop(dependent_type, None)

    def register[
        T
    ](self, dependent_type: type[T] | ty.Callable[..., T], instance: T) -> None:
        if isinstance(dependent_type, type) and issubclass(dependent_type, ty.Protocol):
            instance_type: type[T] = type(instance)
            for base in get_bases(instance_type):
                self._mappings[base] = instance
        else:
            self._mappings[dependent_type] = instance

    def get[
        T
    ](
        self,
        dependent_type: type[T] | ty.Callable[..., T],
        /,
        default: Maybe[T] = MISSING,
    ) -> Maybe[T]:
        return self._mappings.get(dependent_type, default)

    async def close(self, dependent_type: type) -> None:
        instance = self._mappings.pop(dependent_type, None)

        if not instance:
            return

        if is_async_closable(instance):
            await instance.close()
        elif is_closable(instance):
            instance.close()

    async def close_all(self, close_order: ty.Iterable[type]) -> None:
        for type_ in close_order:
            await self.close(type_)


class GraphVisitor[T]:
    __slots__ = ("out_neighbors", "in_neighbors")

    def __init__(self):
        self.out_neighbors: defaultdict[T, list[T]] = defaultdict(list)
        self.in_neighbors: defaultdict[T, list[T]] = defaultdict(list)

    def remove_dependent(self, dependent: T) -> None:
        self.out_neighbors.pop(dependent, None)

    def add_edge(self, dependent: T, dependency: T):
        self.out_neighbors[dependent].append(dependency)
        self.in_neighbors[dependency].append(dependent)

    def get_dependents(self, dependency: T) -> list[T]:
        return self.in_neighbors[dependency]

    def get_dependencies(self, dependent: T) -> list[T]:
        return self.out_neighbors[dependent]

    def find_circular_dependency(self) -> list[T] | None:
        visited: set[T] = set()
        current_path: list[T] = []

        def dfs(node: T) -> list[T] | None:
            if node in current_path:
                # Cycle detected, extract the cycle path
                cycle_start_index = current_path.index(node)
                return current_path[cycle_start_index:] + [node]

            if node in visited:
                return None  # Already processed without cycle

            # Mark the node as visited and add to current path
            visited.add(node)
            current_path.append(node)

            # Visit all dependencies of the node
            for neighbor in self.out_neighbors.get(node, []):
                cycle_path = dfs(neighbor)
                if cycle_path:
                    return cycle_path

            # Remove from current path after processing
            current_path.pop()
            return None

        # Check for cycles in each component of the graph
        for node in self.out_neighbors:
            if node not in visited:
                cycle_path = dfs(node)
                if cycle_path:
                    return cycle_path  # Return the first detected cycle path
        return None

    def clear(self):
        self.out_neighbors.clear()
        self.in_neighbors.clear()
