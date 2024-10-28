import typing as ty

import pytest

from ididi import DependencyGraph
from ididi.errors import ProtocolWithoutFactoryError

dg = DependencyGraph()


class Cache(ty.Protocol):
    def get(self, key: str) -> str: ...


class MemoryCache:
    def get(self, key: str) -> str: ...


@dg.node
class Registry:
    def __init__(self, cache: Cache):
        self.cache = cache


def test_protocols():
    with pytest.raises(ProtocolWithoutFactoryError):
        dg.resolve(Registry)


def test_factory_protocols():
    @dg.node
    def cache_factory() -> Cache:
        return MemoryCache()

    dg.resolve(Registry)
