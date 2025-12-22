from dataclasses import FrozenInstanceError
from inspect import Signature
from typing import Any, Final, Iterable, Literal

from .interfaces import GraphIgnore, GraphIgnoreConfig

EmptyIgnore: Final[tuple[Any]] = tuple()
# DEFAULT_REUSABILITY: bool = False


class FrozenSlot:
    """
    A Mixin class provides a hashable, frozen class with slots defined.
    This is mainly due to the fact that dataclass does not support slots before python 3.10
    """

    __slots__: tuple[str, ...] = ()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return all(
            getattr(self, attr) == getattr(other, attr) for attr in self.__slots__
        )

    def __repr__(self):
        attr_repr = "".join(
            f"{attr}={getattr(self, attr)}, " for attr in self.__slots__
        ).rstrip(", ")
        return f"{self.__class__.__name__}({attr_repr})"

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError("can't set attribute")

    def __hash__(self) -> int:
        attrs = tuple(getattr(self, attr) for attr in self.__slots__)
        return hash(attrs)

class GraphConfig(FrozenSlot):
    __slots__ = ("self_inject", "ignore")

    ignore: GraphIgnore

    def __init__(self, *, self_inject: bool, ignore: GraphIgnoreConfig):

        if not isinstance(ignore, tuple):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = tuple((ignore,))
                else:
                    ignore = tuple(ignore)
            else:
                ignore = (ignore,)

        object.__setattr__(self, "self_inject", self_inject)
        object.__setattr__(self, "ignore", ignore)

try:
    from typing import TypeAliasType  # type: ignore
except ImportError:
    from typing_extensions import TypeAliasType


# DefaultConfig: Final[NodeConfig] = NodeConfig()
CacheMax: Final[int] = 1024
ExtraUnsolvableTypes: Final[tuple[Any, ...]] = (Any, Literal, Signature.empty, TypeAliasType)
DefaultScopeName: Final[str] = "__ididi_default_scope__"

DEFAULT_FACTORY = "default"
FUNCTION_FACTORY = "function"
AFUNCTION_FACTORY = "afunction"
RESOURCE_FACTORY = "resource"
ARESOURCE_FACTORY = "aresource"

FactoryType = Literal["default", "function", "afunction", "resource", "aresource"]

ResolveOrder: Final[dict[FactoryType, int]] = {
    DEFAULT_FACTORY: 1,
    FUNCTION_FACTORY: 2,
    AFUNCTION_FACTORY: 2,
    RESOURCE_FACTORY: 3,
    ARESOURCE_FACTORY: 3,
}
"""
This defines the order of resolving node factory when conflict happens
The idea is that, when a specific factory is provided, then the default node 
would be override
"""
