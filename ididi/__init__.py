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

from ._node import DependentNode as DependentNode
from ._node import Ignore as Ignore
from ._node import Scoped as Scoped
from ._node import use as use
from .api import entry as entry
from .api import resolve as resolve
from .config import IGNORE_PARAM_MARK as IGNORE_PARAM_MARK
from .config import USE_FACTORY_MARK as USE_FACTORY_MARK
from .config import NodeConfig as NodeConfig
from .graph import AsyncScope as AsyncScope
from .graph import Graph as Graph
from .graph import Resolver as Resolver
from .graph import SyncScope as SyncScope
from .interfaces import AsyncResource as AsyncResource
from .interfaces import INode as INode
from .interfaces import INodeConfig as INodeConfig
from .interfaces import Resource as Resource


VERSION="1.6.3"

try:
    import graphviz as graphviz  # type: ignore
except ImportError:
    pass
else:
    from .visual import Visualizer as Visualizer
