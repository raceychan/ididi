# from dataclasses import FrozenInstanceError
from typing import Final, Iterable

from .interfaces import (
    FrozenData,
    GraphIgnore,
    GraphIgnoreConfig,
    NodeIgnore,
    NodeIgnoreConfig,
)


class NodeConfig(FrozenData):

    ignore: NodeIgnore
    reuse: bool

    @classmethod
    def build(cls, ignore: NodeIgnoreConfig = frozenset(), reuse: bool = True):
        if not isinstance(ignore, frozenset):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = frozenset([ignore])
                else:
                    ignore = frozenset(ignore)
            else:
                ignore = frozenset([ignore])

        return cls(reuse=reuse, ignore=ignore)


class GraphConfig(FrozenData):
    self_inject: bool
    ignore: GraphIgnore

    @classmethod
    def build(cls, *, self_inject: bool, ignore: GraphIgnoreConfig):
        if not isinstance(ignore, frozenset):
            if isinstance(ignore, Iterable):
                if isinstance(ignore, str):
                    ignore = frozenset([ignore])
                else:
                    ignore = frozenset(ignore)
            else:
                ignore = frozenset([ignore])

        return cls(self_inject=self_inject, ignore=ignore)


DefaultConfig: Final[NodeConfig] = NodeConfig.build()
CacheMax: Final[int] = 1024
