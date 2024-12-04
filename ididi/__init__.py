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

VERSION = "1.1.1"

__version__ = VERSION

from ._itypes import GraphConfig as GraphConfig
from ._itypes import INode as INode
from ._itypes import INodeConfig as INodeConfig
from ._itypes import NodeConfig as NodeConfig
from .api import entry as entry
from .api import resolve as resolve
from .graph import DependencyGraph as DependencyGraph
from .node import DependentNode as DependentNode
from .node import inject as inject
from .utils.typing_utils import AsyncResource as AsyncResource
from .utils.typing_utils import Resource as Resource

try:
    import graphviz
except ImportError:
    pass
else:
    from .visual import Visualizer as Visualizer
