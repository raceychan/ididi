import typing as ty
from collections import defaultdict
from types import MappingProxyType

from .node import DependentNode
from .utils.typing_utils import is_closable

type NodeDependent[T] = type[T]
"""
### A dependent can be a concrete type or a forward reference
"""

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

type TypeMappings[T] = dict[NodeDependent[T], list[NodeDependent[T]]]
"""
### mapping a type to its dependencies
"""

# TODO: NodeRegistry


class TypeRegistry:
    """
    A mapping of dependent types to their implementations.
    """

    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: TypeMappings[ty.Any] = defaultdict(list)

    def __getitem__[T](self, dependent_type: type[T]) -> list[type[T]]:
        return self._mappings[dependent_type].copy()

    def __contains__(self, dependent_type: type) -> bool:
        return dependent_type in self._mappings

    def register(self, dependent_type: type):
        """
        Register a dependent type and update its base types.
        """
        self._mappings[dependent_type].append(dependent_type)

        # __mro__[1:-1] skip dependent itself and object
        for base in dependent_type.__mro__[1:-1]:
            self._mappings[base].append(dependent_type)

    def remove(self, dependent_type: type):
        """
        Remove a dependent type and update its base types.
        """

        for base in dependent_type.__mro__[1:-1]:
            self._mappings[base].remove(dependent_type)

        del self._mappings[dependent_type]

    def get(
        self, dependent_type: type, /, default: list[type] | None = None
    ) -> list[type] | None:
        return self._mappings.get(dependent_type, default)

    def clear(self) -> None:
        self._mappings.clear()


class ResolutionRegistry:
    """
    A mapping of dependent types to their resolved instances.
    """

    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: ResolvedInstances[ty.Any] = {}

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, dependent_type: type | ty.Callable[..., ty.Any]) -> bool:
        return dependent_type in self._mappings

    def __getitem__[T](self, dependent_type: type[T] | ty.Callable[..., T]) -> T:
        return self._mappings[dependent_type]

    def remove(self, dependent_type: type) -> None:
        self._mappings.pop(dependent_type, None)

    def clear(self) -> None:
        self._mappings.clear()

    def register[
        T
    ](self, dependent_type: type[T] | ty.Callable[..., T], instance: T) -> None:
        self._mappings[dependent_type] = instance

    async def close(self, dependent_type: type) -> None:
        instance = self._mappings.pop(dependent_type, None)
        if instance and is_closable(instance):
            await instance.close()

    async def close_all(self, close_order: ty.Iterable[type]) -> None:
        for type_ in close_order:
            await self.close(type_)
