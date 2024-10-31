# from abc import ABC
# import contextlib

"""
BUG: importing typing would cause
any class that defines both __aenter__ and __aexit__
to be considered as an subclass of ABC.
"""

# from typing import GenericAlias
# import contextlib
from abc import ABC
import _collections_abc

# class AbstractAsyncContextManager(ABC):
#     @classmethod
#     def __subclasshook__(cls, C):
#         if cls is AbstractAsyncContextManager:
#             return _collections_abc._check_methods(C, "__aenter__",
#                                                    "__aexit__")
#         return NotImplemented



class ArbitraryResource:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass



print(f"is abc.ABC subclass: {issubclass(ArbitraryResource, ABC)}")

try:
    assert not issubclass(ArbitraryResource, ABC)
except Exception as e:
    print(ArbitraryResource.__bases__)
    raise e
