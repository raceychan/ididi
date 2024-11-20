import typing as ty
from collections import defaultdict
from types import MappingProxyType

from ._type_resolve import get_bases
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

    def update(self, other: "TypeRegistry"):
        self._mappings.update(other._mappings)

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

    def update(self, other: "ResolutionRegistry"):
        self._mappings.update(other._mappings)

    def register[
        T
    ](self, dependent_type: type[T] | ty.Callable[..., T], instance: T) -> None:
        if isinstance(dependent_type, type):
            instance_type: type[T] = type(instance)
            for base in get_bases(instance_type):
                if base in self._mappings:
                    continue
                self._mappings[base] = instance
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
