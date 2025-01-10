"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

from datetime import datetime, timezone
from typing import Annotated, TypeVar

from ididi import DependencyGraph, use
from ididi._type_resolve import IDIDI_INJECT_IGNORE_MARK
from tests.test_data import UserService

# T = TypeVar("T")

# Ignore = Annotated[T, IDIDI_INJECT_IGNORE_MARK]


# def test_ignore_param():
#     class User:
#         def __init__(self, name: Ignore[str] = "3"):
#             self._name = name

#     dg = DependencyGraph()

#     dg.static_resolve(User)
#     # node = dg.nodes[User]


