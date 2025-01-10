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

VERSION = "1.2.3"

__version__ = VERSION

from .api import entry as entry
from .api import resolve as resolve
from .graph import AsyncScope as AsyncScope
from .graph import DependencyGraph as DependencyGraph
from .graph import SyncScope as SyncScope
from .interfaces import INode as INode
from .interfaces import INodeConfig as INodeConfig
from .node import DependentNode as DependentNode
from .node import Ignore as Ignore
from .node import NodeConfig as NodeConfig
from .node import use as use
from .utils.typing_utils import AsyncResource as AsyncResource
from .utils.typing_utils import Resource as Resource

try:
    import graphviz as graphviz  # type: ignore
except ImportError:
    pass
else:
    from .visual import Visualizer as Visualizer
