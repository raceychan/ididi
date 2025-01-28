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
        return (self.ignore == other.ignore) and (self.reuse == other.reuse)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.reuse=}, {self.ignore=})"

    def __setattr__(self, name: str, value: Any) -> None:
        raise FrozenInstanceError("can't set attribute")

    def __hash__(self) -> int:
        return hash((self.reuse, self.ignore))


class GraphConfig:
    __slots__ = ("self_inject", "ignore")

    def __init__(self, *, self_inject: bool, ignore: GraphIgnoreConfig):
        self.self_inject = self_inject
        if not isinstance(ignore, frozenset):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = frozenset([ignore])
                else:
                    ignore = frozenset(ignore)
            else:
                ignore = frozenset([ignore])

        self.ignore = ignore


DefaultConfig = NodeConfig()
