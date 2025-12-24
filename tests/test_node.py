import inspect
from types import MethodType
from typing import Annotated

import pytest

from ididi import DependentNode, resolve_meta, use
from ididi._node import NodeMeta, build_dependencies
from ididi.config import EmptyIgnore
from ididi.errors import NotSupportedError
from ididi.utils.param_utils import MISSING
from ididi.utils.typing_utils import flatten_annotated


class A:
    def __init__(self, a: int = 5):
        self.a = a


class B:
    def __init__(self, b: str = "b"):
        self.b = b


class Config:
    def __init__(self, env: str = "dev"):
        self.env = env


class Database:
    def __init__(self, config: Config):
        self.config = config


class Service:
    def __init__(self, db: Database):
        self.db = db


class ComplexDependency:
    def __init__(self, a: A, /, b: B, *args: int, config: Config, **kwargs: str):
        self.a = a
        self.b = b
        self.args = args  # This should be a tuple
        self.config = config
        self.kwargs = kwargs


def complex_factory(a: A, b: B, config: Config) -> ComplexDependency:
    return ComplexDependency(a, b, 1, 2, 3, config=config, extra="test")


def test_typed_annotation():
    # Test forward reference handling
    class ServiceA:
        def __init__(self, nums: list[int]):
            self.nums = nums

    node = DependentNode.from_node(ServiceA)
    ser_dep = node.get_param("nums")
    print(ser_dep)


def test_empty_init():
    class EmptyService:
        pass

    node = DependentNode.from_node(EmptyService)
    assert not node.dependencies


def test_varidc_keyword_args():
    class ARGDep:
        def __init__(self, *nums: int):
            self.nums = nums

    class KWDep:
        def __init__(self, **kwargs: int):
            self.kwargs = kwargs

    node = DependentNode.from_node(KWDep)
    node2 = DependentNode.from_node(ARGDep)
    assert not node.dependencies
    assert not node2.dependencies


def test_node_config_frozen():
    node = DependentNode.from_node(Service)
    with pytest.raises(AttributeError):
        node.reuse = False


def test_node_from_annotated():
    def hello() -> Annotated[str, "hello"]:
        return "hello"

    node = DependentNode.from_node(hello)
    assert node.dependent is str


def test_node_with_not_supported_type():
    with pytest.raises(NotSupportedError):
        DependentNode.from_node(dict(a=1, b=2))


def test_node_meta_rejects_reuse_and_ignore():
    with pytest.raises(NotSupportedError):
        NodeMeta(reuse=True, ignore=True)


def test_resolve_use_accepts_flattened_metadata_list_with_factory():
    def factory() -> str:
        return "value"

    annotation = Annotated[str, use(factory, reuse=True)]
    flattened = flatten_annotated(annotation)
    node_meta = resolve_meta(flattened)

    assert isinstance(node_meta, NodeMeta)
    assert node_meta.factory is factory
    assert node_meta.reuse is True


def test_resolve_use_accepts_flattened_metadata_list_without_factory():
    annotation = Annotated[str, use()]
    flattened = flatten_annotated(annotation)
    node_meta = resolve_meta(flattened)

    assert isinstance(node_meta, NodeMeta)
    assert node_meta.factory is MISSING
    assert node_meta.reuse is MISSING


def test_build_dependencies_skips_bound_instance_self():
    class Builder:
        def method(self, value: int) -> int:
            return value

    bound = MethodType(Builder.method, Builder())
    deps = build_dependencies(bound, inspect.signature(bound))
    assert deps == []


def test_get_param_missing_raises():
    class NeedsConfig:
        def __init__(self, config: Config):
            self.config = config

    node = DependentNode._from_class(NeedsConfig, reuse=MISSING, ignore=EmptyIgnore)
    with pytest.raises(AttributeError):
        node.get_param("unknown")


def test_from_node_normalizes_ignore_configs():
    class Sample:
        def __init__(self, name: str):
            self.name = name

    default_ignore = DependentNode.from_node(Sample, ignore=MISSING)
    assert default_ignore.ignore == tuple()

    with_string = DependentNode.from_node(Sample, ignore="name")
    assert with_string.ignore == ("name",)
