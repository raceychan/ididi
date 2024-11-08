"""
IDIDI
~~~~~~~~~~~~~~~~~~~~~

Ididi is a pythonic dependency injection lib, with ergonomics api, without boilplate code, works out of the box.



   >>> import ididi
   >>> service = ididi.solve(MyService)
   >>> assert isinstance(service, MyService)

is at <https://requests.readthedocs.io>.

:copyright: (c) 2024 by race chan.
:license: MIT, see LICENSE for more details.
"""

VERSION = "0.2.3"

__version__ = VERSION

from .api import entry as entry, solve as solve
from .graph import DependencyGraph as DependencyGraph
from .node import DependentNode as DependentNode
from .visual import Visualizer as Visualizer
