"""
IDIDI
~~~~~~~~~~~~~~~~~~~~~

Ididi is a pythonic dependency injection lib, with ergonomic apis, without boilplate code, works out of the box.

>>> import ididi
>>> service = ididi.resolve(MyService)
>>> assert isinstance(service, MyService)

source code is at <https://github.com/raceychan/ididi>.

copyright: (c) 2024 by race chan.
license: MIT, see LICENSE for more details.
"""

from typing import Annotated as Annotated

from ._node import DependentNode as DependentNode
from ._node import Ignore as Ignore
from ._node import NodeMeta as NodeMeta
from ._node import Scoped as Scoped
from ._node import resolve_meta as resolve_meta
from ._node import use as use
from .api import entry as entry
from .api import resolve as resolve
from .graph import AsyncScope as AsyncScope
from .graph import Graph as Graph
from .graph import Resolver as Resolver
from .graph import SyncScope as SyncScope
from .interfaces import AsyncResource as AsyncResource
from .interfaces import INode as INode
from .interfaces import Resource as Resource
from .utils.param_utils import is_provided as is_provided

VERSION="1.8.3"

try:
    import graphviz as graphviz  # type: ignore
except ImportError:
    pass
else:
    from .visual import Visualizer as Visualizer
