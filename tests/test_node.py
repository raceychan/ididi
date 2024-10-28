import typing as ty

import pytest

from ididi.errors import GenericDependencyNotSupportedError
from ididi.node import DependencyNode


def print_dependency_tree(node: DependencyNode[ty.Any], level: int = 0):
    indent = "  " * level
    print(f"{indent} {node.dependent=}, {node.default=}")
    for dep in node.dependency_params:
        print_dependency_tree(dep.dependency, level + 1)


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


class GenericService[T]:
    def __init__(self, item: T):
        self.item = item


@pytest.fixture
def basic_nodes():
    return {
        "A": DependencyNode.from_node(A),
        "B": DependencyNode.from_node(B),
        "Config": DependencyNode.from_node(Config),
    }


# def test_nested_dependencies():
#     config_node = DependencyNode.from_node(Config)
#     db_node = DependencyNode.from_node(Database)
#     db_node.dependencies = [config_node]
#     service_node = DependencyNode.from_node(Service)
#     service_node.dependencies = [db_node]

#     service = service_node.build()

#     assert isinstance(service, Service)
#     assert isinstance(service.db, Database)
#     assert isinstance(service.db.config, Config)
#     assert service.db.config.env == "dev"


# def test_complex_dependency_injection():
#     complex_node = DependencyNode.from_node(ComplexDependency)

#     complex_obj = complex_node.build()

#     assert isinstance(complex_obj, ComplexDependency)
#     assert isinstance(complex_obj.a, A)
#     assert complex_obj.a.a == 5
#     assert isinstance(complex_obj.b, B)
#     assert complex_obj.b.b == "b"
#     assert complex_obj.args == ()  # Empty tuple, as no args were provided
#     assert isinstance(complex_obj.config, Config)
#     assert complex_obj.config.env == "dev"
#     assert complex_obj.kwargs == {}


def test_unsolvable_dependency():
    DependencyNode.from_node(ComplexDependency)


def test_factory_function():
    factory_node = DependencyNode.from_node(complex_factory)

    factory_obj = factory_node.build()

    assert isinstance(factory_obj, ComplexDependency)
    assert isinstance(factory_obj.a, A)
    assert factory_obj.a.a == 5
    assert isinstance(factory_obj.b, B)
    assert factory_obj.b.b == "b"
    assert factory_obj.args == (1, 2, 3)
    assert isinstance(factory_obj.config, Config)
    assert factory_obj.config.env == "dev"
    assert factory_obj.kwargs == {"extra": "test"}


def test_generic_service_not_supported():
    with pytest.raises(GenericDependencyNotSupportedError):
        DependencyNode.from_node(GenericService[str])


# def test_forward_reference():
#     # Test forward reference handling
#     class ServiceA:
#         def __init__(self, b: "ServiceB"):
#             self.b = b

#     class ServiceB:
#         def __init__(self):
#             pass

#     node = DependencyNode.from_node(ServiceA)
#     assert any(
#         isinstance(dep.dependency, ForwardDependency) for dep in node.dependency_params
#     )


def test_empty_init():
    # Test class with no __init__
    class EmptyService:
        pass

    node = DependencyNode.from_node(EmptyService)
    instance = node.build()
    assert isinstance(instance, EmptyService)
    assert not node.dependency_params


def test_factory_without_return_type():
    # Test factory without return type annotation
    def bad_factory():
        return object()

    with pytest.raises(ValueError, match="Factory must have a return type"):
        DependencyNode.from_node(bad_factory)
