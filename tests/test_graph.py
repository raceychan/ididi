import inspect
from abc import ABC, abstractmethod

import pytest

from ididi.errors import (
    ABCWithoutImplementationError,
    MultipleImplementationsError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
)
from ididi.graph import DependencyGraph, is_closable

dag = DependencyGraph()


@dag.node
class Config:
    def __init__(self, env: str = "prod"):
        self.env = env


class Database:
    def __init__(self, config: Config):
        self.config = config


class Cache:
    def __init__(self, config: Config):
        self.config = config


class UserRepository:
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache


class AuthService:
    def __init__(self, db: Database):
        self.db = db


@dag.node(reuse=False)
class UserService:
    def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
        self.repo = repo
        self.auth = auth


@dag.node(reuse=False)
def auth_service_factory(database: Database) -> AuthService:
    return AuthService(db=database)


class Repository(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the repository data."""
        pass


def test_dag_resolve():
    service = dag.resolve(UserService)
    assert isinstance(service.repo.db, Database)
    assert isinstance(service.repo.cache, Cache)
    assert isinstance(service.auth.db, Database)


def test_get_dependency_types():
    database_dependencies = dag.get_dependency_types(Database)
    assert len(database_dependencies) == 1
    assert Config in database_dependencies


def test_get_dependent_types():
    database_dependents = dag.get_dependent_types(Database)
    assert len(database_dependents) == 2
    assert UserRepository in database_dependents
    assert AuthService in database_dependents


def test_top_level_builtin_dependency():
    with pytest.raises(TopLevelBulitinTypeError):
        dag.resolve(int)


def test_missing_implementation():
    dag = DependencyGraph()
    with pytest.raises(ABCWithoutImplementationError):
        dag.resolve(Repository)


def test_multiple_implementations():
    dag = DependencyGraph()

    @dag.node
    class Repo1(Repository):  # type: ignore
        def save(self) -> None:
            pass

    @dag.node
    class Repo2(Repository):  # type: ignore
        def save(self) -> None:
            pass

    # @dag.node
    # def repo_factory() -> Repository:
    #     return Repo2()

    with pytest.raises(MultipleImplementationsError):
        dag.resolve(Repository)


def test_multiple_implementations_with_factory():
    dag = DependencyGraph()

    @dag.node
    class Repo1(Repository):  # type: ignore
        def save(self) -> None:
            pass

    @dag.node
    class Repo2(Repository):  # type: ignore
        def save(self) -> None:
            pass

    @dag.node
    def repo_factory() -> Repository:
        return Repo2()

    repo = dag.resolve(Repository)
    assert isinstance(repo, Repo2)


def test_resource_cleanup():
    dag = DependencyGraph()
    closed_resources: set[str] = set()

    @dag.node
    class Resource1:
        async def close(self):
            closed_resources.add("resource1")

    @dag.node
    class Resource2:
        def __init__(self, r1: Resource1):
            self.r1 = r1

        async def close(self):
            closed_resources.add("resource2")

    # Resolve and verify resources
    instance = dag.resolve(Resource2)
    assert isinstance(instance, Resource2)

    # Test cleanup
    import asyncio

    asyncio.run(dag.aclose())
    assert closed_resources == {"resource1", "resource2"}


def test_node_removal():
    dag = DependencyGraph()

    @dag.node
    class Service:
        pass

    node = dag.nodes[Service]
    dag.remove_node(node)

    assert Service not in dag.nodes

    dag.resolve(Service)


def test_graph_reset():
    dag = DependencyGraph()

    @dag.node
    class Service:
        def __init__(self, name: str = "test"):
            self.name = name

    # First resolution
    instance1 = dag.resolve(Service)
    assert instance1.name == "test"

    # Reset and resolve again
    dag.reset()
    instance2 = dag.resolve(Service)

    # Verify we got a new instance
    assert instance1 is not instance2


def test_graph_repr():
    dag = DependencyGraph()

    @dag.node
    class LeafService:
        pass

    @dag.node
    class RootService:  #  type: ignore
        def __init__(self, leaf: LeafService):
            self.leaf = leaf

    repr_str = dag.__stats__()
    assert "nodes=2" in repr_str
    assert "resolved=0" in repr_str
    assert "RootService" in repr_str
    assert "LeafService" in repr_str

    str(dag)


def test_node_removal_cleanup():
    dag = DependencyGraph()

    @dag.node
    class Dependency:
        pass

    @dag.node
    class Service:
        def __init__(self, dep: Dependency):
            self.dep = dep

    dag.resolve(Service)

    # Get the node and remove it
    node = dag.nodes[Dependency]
    dag.remove_node(node)

    # Check that all references are cleaned up
    assert Dependency not in dag.nodes
    assert Dependency not in dag.type_mappings[Dependency]
    dependents = dag.get_dependent_types(Dependency)


def test_factory_override():
    dag = DependencyGraph()

    @dag.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # First resolution
    service1 = dag.resolve(Service)
    assert service1.name == "default"

    # Override with factory
    @dag.node
    def service_factory() -> Service:
        return Service(name="overridden")

    # Verify factory registration
    node = dag.nodes[Service]
    assert node.factory == service_factory, "Factory was not properly registered"
    assert inspect.isfunction(node.factory), "Factory should be a function"

    # # Clear any cached instances
    # dag.reset()

    # Second resolution should use the factory
    service2 = dag.resolve(Service)
    assert service2.name == "overridden"


def test_factory_registration():
    dag = DependencyGraph()

    # First register class
    @dag.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # Then register factory
    call_count = 0

    @dag.node
    def service_factory() -> Service:
        nonlocal call_count
        call_count += 1
        return Service(name=f"factory_{call_count}")

    # Verify factory is registered
    node = dag.nodes[Service]
    assert node.factory == service_factory, "Factory should override class constructor"

    # Verify factory is called during resolution
    service = dag.resolve(Service)
    assert call_count == 1, "Factory should be called during resolution"
    assert service.name == "factory_1", "Factory-created instance should be used"


def test_unsupported_annotation():
    dag = DependencyGraph()

    with pytest.raises(UnsolvableDependencyError):

        @dag.node
        class BadService:
            def __init__(self, bad: None):  # object is not a proper annotation
                self.bad = bad

        dag.resolve(BadService)


def test_resource_type_check():
    dag = DependencyGraph()

    class NonResource:
        pass

    class Resource:
        async def close(self):
            pass

    assert not is_closable(NonResource())
    assert is_closable(Resource())


def test_initialization_order():
    dag = DependencyGraph()

    @dag.node
    class ServiceC:
        pass

    @dag.node
    class ServiceB:
        def __init__(self, c: ServiceC):
            self.c = c

    @dag.node
    class ServiceA:
        def __init__(self, b: ServiceB):
            self.b = b

    order = dag.get_initialization_order()
    # C should be initialized before B, and B before A
    assert order.index(ServiceC) < order.index(ServiceB)
    assert order.index(ServiceB) < order.index(ServiceA)


def test_dependency_override():
    dag = DependencyGraph()

    @dag.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # Override with kwargs
    instance = dag.resolve(Service, name="overridden")
    assert instance.name == "overridden"


def test_nested_dependency_override():
    dag = DependencyGraph()

    @dag.node
    class Inner:
        def __init__(self, value: str = "inner"):
            self.value = value

    @dag.node
    class Outer:
        def __init__(self, inner: Inner):
            self.inner = inner

    # Override nested dependency
    instance = dag.resolve(Outer, inner=Inner(value="overridden"))
    assert instance.inner.value == "overridden"


def test_abstract_base_resolution():
    dg = DependencyGraph()

    class AbstractService(ABC):
        @abstractmethod
        def get_name(self) -> str:
            pass

    @dg.node
    class ConcreteService(AbstractService):
        def get_name(self) -> str:
            return "concrete"

    instance = dg.resolve(AbstractService)
    assert isinstance(instance, ConcreteService)
    assert instance.get_name() == "concrete"


def test_multiple_dependency_paths():
    dag = DependencyGraph()

    @dag.node
    class Shared:
        def __init__(self, value: str = "shared"):
            self.value = value

    @dag.node
    class Service1:
        def __init__(self, shared2: Shared):
            self.shared2 = shared2

    @dag.node
    class Service2:
        def __init__(self, shared1: Shared):
            self.shared1 = shared1

    @dag.node
    class Root:
        def __init__(self, s1: Service1, s2: Service2):
            self.s1 = s1
            self.s2 = s2

    instance = dag.resolve(Root)
    assert dag.resolution_registry[Shared] is instance.s1.shared2
    assert dag.resolution_registry[Shared] is instance.s2.shared1
    # Verify shared instance is actually shared
    assert instance.s1.shared2 is instance.s2.shared1
    assert instance.s1.shared2.value == instance.s2.shared1.value == "shared"


@pytest.mark.debug
def test_type_mapping_cleanup():
    dag = DependencyGraph()

    class Interface(ABC):
        pass

    @dag.node
    class Implementation(Interface):
        pass

    # Verify type mapping
    assert Implementation in dag.type_mappings[Interface]

    # Remove implementation
    node = dag.nodes[Implementation]
    dag.remove_node(node)

    # Verify type mapping is cleaned up
    assert Interface not in dag.type_mappings or not dag.type_mappings[Interface]
