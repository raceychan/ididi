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

VERSION = "1.0.5"

__version__ = VERSION

from ._itypes import GraphConfig as GraphConfig
from ._itypes import INode as INode
from ._itypes import NodeConfig as NodeConfig
from .api import entry as entry
from .api import resolve as resolve
from .graph import DependencyGraph as DependencyGraph
from .node import DependentNode as DependentNode

try:
    import graphviz
except ImportError:
    pass
else:
    from .visual import Visualizer as Visualizer
