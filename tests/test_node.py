from typing import Annotated

import pytest

from ididi import DependentNode


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
    ser_dep = node.dependencies["nums"]
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
        node.config.reuse = False


def test_node_from_annotated():
    def hello() -> Annotated[str, "hello"]:
        return "hello"

    node = DependentNode.from_node(hello)
    assert node.dependent is str
