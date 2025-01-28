from dataclasses import FrozenInstanceError
from typing import Any, Iterable

from .interfaces import GraphIgnoreConfig, NodeIgnore, NodeIgnoreConfig


class NodeConfig:
    __slots__ = ("reuse", "ignore")

    ignore: NodeIgnore
    reuse: bool

    def __init__(
        self,
        *,
        reuse: bool = True,
        ignore: NodeIgnoreConfig = frozenset(),
    ):

        if not isinstance(ignore, frozenset):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = frozenset([ignore])
                else:
                    ignore = frozenset(ignore)
            else:
                ignore = frozenset([ignore])

        object.__setattr__(self, "ignore", ignore)
        object.__setattr__(self, "reuse", reuse)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NodeConfig):
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


class GraphConfig(NodeConfig):
    __slots__ = ("self_inject", "ignore")

    def __init__(self, *, self_inject: bool, ignore: GraphIgnoreConfig):

        if not isinstance(ignore, frozenset):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = frozenset([ignore])
                else:
                    ignore = frozenset(ignore)
            else:
                ignore = frozenset([ignore])

        object.__setattr__(self, "self_inject", self_inject)
        object.__setattr__(self, "ignore", ignore)


DefaultConfig = NodeConfig()
