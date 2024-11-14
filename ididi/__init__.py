"""
IDIDI
~~~~~~~~~~~~~~~~~~~~~

Ididi is a pythonic dependency injection lib, with ergonomic apis, without boilplate code, works out of the box.

>>> import ididi
>>> service = ididi.solve(MyService)
>>> assert isinstance(service, MyService)

source code is at <https://github.com/raceychan/ididi>.

copyright: (c) 2024 by race chan.
license: MIT, see LICENSE for more details.
"""

VERSION = "0.2.9"

__version__ = VERSION

from .api import entry as entry
from .api import solve as solve
from .graph import DependencyGraph as DependencyGraph
from .node import DependentNode as DependentNode
from .types import INodeConfig as INodeConfig
from .visual import Visualizer as Visualizer
