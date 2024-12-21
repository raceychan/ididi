import inspect
import sys
import typing as ty
from abc import ABC, abstractmethod
from unittest import mock

import pytest

from ididi.errors import (
    ABCNotImplementedError,
    AsyncResourceInSyncError,
    GenericDependencyNotSupportedError,
    MergeWithScopeStartedError,
    MissingAnnotationError,
    PositionalOverrideError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
)
from ididi.graph import DependencyGraph

T = ty.TypeVar("T")


def test_import_graphviz_failure():
    with mock.patch.dict(sys.modules, {"graphviz": None}):
        # Remove cached ididi module
        if "ididi" in sys.modules:
            del sys.modules["ididi"]

        # Import ididi after patching
        import ididi

        # Ensure Visualizer is not available
        assert not hasattr(
            ididi, "Visualizer"
        ), "Visualizer should not be available when graphviz is missing"


# Replace global dag with a fixture
@pytest.fixture(scope="module")
def dg():
    return DependencyGraph()


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


class UserService:
    def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
        self.repo = repo
        self.auth = auth


def auth_service_factory(database: Database) -> AuthService:
    return AuthService(db=database)


def test_dag_resolve(dg: DependencyGraph):
    @dg.node(reuse=False)
    class UserService:
        def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
            self.repo = repo
            self.auth = auth

    service = dg.resolve(UserService)
    assert isinstance(service.repo.db, Database)
    assert isinstance(service.repo.cache, Cache)
    assert isinstance(service.auth.db, Database)


def test_get_dependency_types(dg: DependencyGraph):
    database_dependencies = dg.visitor.get_dependencies(Database)
    assert len(database_dependencies) == 1
    assert Config in database_dependencies


def test_get_dependent_types(dg: DependencyGraph):
    database_dependents = dg.visitor.get_dependents(Database)
    assert len(database_dependents) == 2
    assert UserRepository in database_dependents
    assert AuthService in database_dependents


def test_node_signature_change_after_factory(dg: DependencyGraph):
    class UserService:
        def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
            self.repo = repo
            self.auth = auth

    dg.static_resolve(UserService)
    node = dg.nodes[UserService]
    assert len(node.signature.dprams) == 3

    @dg.node
    def user_service_factory(repo: UserRepository, auth: AuthService) -> UserService:
        return UserService(repo, auth)

    node = dg.nodes[UserService]
    assert len(node.signature.dprams) == 2


def test_top_level_builtin_dependency(dg: DependencyGraph):
    with pytest.raises(TopLevelBulitinTypeError):
        dg.resolve(int)


def test_missing_implementation(dg: DependencyGraph):
    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    with pytest.raises(ABCNotImplementedError):
        dg.resolve(Repository)


def test_multiple_implementations(dg: DependencyGraph):
    dg.reset(clear_nodes=True)

    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    @dg.node
    class Repo1(Repository):
        def save(self) -> None:
            pass

    @dg.node
    class Repo2(Repository):
        def save(self) -> None:
            pass

    dg.resolve(Repository)


def test_multiple_implementations_with_factory(dg: DependencyGraph):
    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    @dg.node
    class Repo1(Repository):
        def save(self) -> None:
            pass

    @dg.node
    class Repo2(Repository):
        def save(self) -> None:
            pass

    @dg.node
    def repo_factory() -> Repository:
        return Repo1()

    assert Repository in dg.nodes

    repo = dg.resolve(Repository)
    assert isinstance(repo, Repo1)


def test_static_resolve_factory(dg: DependencyGraph):
    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    @dg.node
    class Repo1(Repository):
        def save(self) -> None:
            pass

    @dg.node
    class Repo2(Repository):
        def save(self) -> None:
            pass

    @dg.node
    def repo_factory() -> Repository:
        return Repo1()

    assert Repository in dg.nodes

    repo = dg.static_resolve(repo_factory)
    assert Repository in dg.type_registry[Repository]


@pytest.mark.asyncio
async def test_resource_cleanup(dg: DependencyGraph):
    closed_resources: set[str] = set()

    @dg.node
    class Resource1:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.close()

        async def close(self):
            closed_resources.add("resource1")

    @dg.node
    class Resource2:
        def __init__(self, r1: Resource1):
            self.r1 = r1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await self.close()

        async def close(self):
            closed_resources.add("resource2")

    # Resolve and verify resources
    with pytest.raises(AsyncResourceInSyncError):
        with dg.scope() as scope:
            instance = scope.resolve(Resource2)

    async with dg.scope() as scope:
        instance = await scope.resolve(Resource2)
        assert isinstance(instance, Resource2)

    assert closed_resources == {"resource1", "resource2"}


def test_node_removal(dg: DependencyGraph):
    @dg.node
    class Service:
        pass

    node = dg.nodes[Service]
    dg.remove_node(node)

    assert Service not in dg.nodes

    dg.resolve(Service)


def test_graph_reset(dg: DependencyGraph):
    @dg.node
    class Service:
        def __init__(self, name: str = "test"):
            self.name = name

    # First resolution
    instance1 = dg.resolve(Service)
    assert instance1.name == "test"

    # Reset and resolve again
    dg.reset()
    instance2 = dg.resolve(Service)

    # Verify we got a new instance
    assert instance1 is not instance2


def test_node_removal_cleanup(dg: DependencyGraph):
    dg.reset(clear_nodes=True)

    @dg.node
    class Dependency:
        pass

    @dg.node
    class Service:
        def __init__(self, dep: Dependency):
            self.dep = dep

    dg.resolve(Service)

    # Get the node and remove it
    node = dg.nodes[Dependency]
    dg.remove_node(node)

    # Check that all references are cleaned up
    assert Dependency not in dg.nodes
    assert Dependency not in dg.type_registry[Dependency]
    assert Dependency not in dg.resolution_registry
    assert Dependency not in dg.resolved_nodes


def test_facwry_override(dg: DependencyGraph):
    @dg.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # First resolution
    service1 = dg.resolve(Service)
    assert service1.name == "default"

    # Override with factory
    @dg.node
    def service_factory() -> Service:
        return Service(name="overridden")

    # Verify factory registration
    node = dg.nodes[Service]
    assert node.factory == service_factory, "Factory was not properly registered"
    assert inspect.isfunction(node.factory), "Factory should be a function"

    # Clear any cached instances
    dg.reset()

    # Second resolution should use the factory
    service2 = dg.resolve(Service)
    assert service2.name == "overridden"


def test_factory_registration(dg: DependencyGraph):
    # First register class
    @dg.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # Then register factory
    call_count = 0

    @dg.node
    def service_factory() -> Service:
        nonlocal call_count
        call_count += 1
        return Service(name=f"factory_{call_count}")

    # Verify factory is registered
    node = dg.nodes[Service]
    assert node.factory == service_factory, "Factory should override class constructor"

    # Verify factory is called during resolution
    service = dg.resolve(Service)
    assert call_count == 1, "Factory should be called during resolution"
    assert service.name == "factory_1", "Factory-created instance should be used"


def test_unsupported_annotation(dg: DependencyGraph):
    @dg.node
    class BadService:
        def __init__(self, bad: None):  # object is not a proper annotation
            self.bad = bad

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(BadService)

    class New:
        def __init__(self, name: str): ...

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(New)

    class Old:
        def __init__(self, age: int = 3): ...

    @dg.node
    def new_factory(age: int) -> New:
        return New(age)

    with pytest.raises(UnsolvableDependencyError):
        dg.resolve(New)


def test_dependency_override(dg: DependencyGraph):
    @dg.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # Override with kwargs
    instance = dg.resolve(Service, name="overridden")
    assert instance.name == "overridden"


def test_nested_dependency_override(dg: DependencyGraph):
    @dg.node
    class Inner:
        def __init__(self, value: str = "inner"):
            self.value = value

    @dg.node
    class Outer:
        def __init__(self, inner: Inner):
            self.inner = inner

    # Override nested dependency
    instance = dg.resolve(Outer, inner=Inner(value="overridden"))
    assert instance.inner.value == "overridden"


def test_abstract_base_resolution(dg: DependencyGraph):
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


def test_multiple_dependency_paths(dg: DependencyGraph):
    class Shared:
        def __init__(self, value: str = "shared"):
            self.value = value

    class Service1:
        def __init__(self, shared2: Shared):
            self.shared2 = shared2

    class Service2:
        def __init__(self, shared1: Shared):
            self.shared1 = shared1

    class Root:
        def __init__(self, s1: Service1, s2: Service2):
            self.s1 = s1
            self.s2 = s2

    instance = dg.resolve(Root)
    assert dg.resolution_registry[Shared] is instance.s1.shared2
    assert dg.resolution_registry[Shared] is instance.s2.shared1
    # Verify shared instance is actually shared
    assert instance.s1.shared2 is instance.s2.shared1
    assert instance.s1.shared2.value == instance.s2.shared1.value == "shared"


def test_type_mapping_cleanup(dg: DependencyGraph):
    class Interface(ABC):
        pass

    @dg.node
    class Implementation(Interface):
        pass

    # Verify type mapping
    assert Implementation in dg.type_registry[Interface]

    # Remove implementation
    node = dg.nodes[Implementation]
    dg.remove_node(node)

    # Verify type mapping is cleaned up
    assert Interface not in dg.type_registry or not dg.type_registry[Interface]


@pytest.mark.asyncio
async def test_graph_without_static_resolve(dg: DependencyGraph):
    # This test specifically needs a new dag instance
    dg = DependencyGraph()

    @dg.node(reuse=False)
    class UserService:
        def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
            self.repo = repo
            self.auth = auth

    dg.resolve(UserService)

    with pytest.raises(PositionalOverrideError):
        dg.resolve(UserRepository, 1)

    with pytest.raises(PositionalOverrideError):
        await dg.aresolve(UserRepository, 1)

    with pytest.raises(PositionalOverrideError):
        dg.resolve(UserRepository, 1, 2)

    with pytest.raises(PositionalOverrideError):
        await dg.aresolve(UserRepository, 1, 2)

    await dg.aresolve(UserRepository, db="asdf")


def test_graph_replace_node(dg: DependencyGraph):
    @dg.node
    class Service:
        pass

    node = dg.nodes[Service]

    dg.replace_node(node, node)


def test_resolve_node_without_annotation():
    dag = DependencyGraph()

    class Config:
        def __init__(self, a):
            self.a = a

    class Service:
        def __init__(self, config: Config):
            self.config = config

    with pytest.raises(MissingAnnotationError):
        dag.resolve(Service)

    @dag.node
    def config_factory() -> Config:
        return Config(a=1)

    service = dag.resolve(Service)
    print(dag.nodes[Config])
    assert service.config.a == 1


class GenericService(ty.Generic[T]):
    def __init__(self, item: T):
        self.item = item


def test_generic_service_not_supported(dg: DependencyGraph):
    with pytest.raises(GenericDependencyNotSupportedError):
        dg.resolve(GenericService[str])


def test_generic_service_with_default(dg: DependencyGraph):
    class GenericService(ty.Generic[T]):
        def __init__(self, item: T = "5"):
            self.item = item

    # this should fail, since it is confusing behavior
    with pytest.raises(GenericDependencyNotSupportedError):
        dg.resolve(GenericService[str])

    repr(dg)


def test_node_config_non_transitive(dg: DependencyGraph):
    class Base:
        def __init__(self, name: str = "base"): ...

    class Sub:
        def __init__(self, b: Base): ...

    dg.node(reuse=False)(Sub)
    dg.static_resolve(Sub)

    assert dg.nodes[Base].config.reuse == True


def test_partial_node(dg: DependencyGraph):
    class Base:
        def __init__(self, name: str = "base"): ...

    class Sub:
        def __init__(self, b: Base, age: int):
            self.b = b
            self.age = age

    with pytest.raises(UnsolvableDependencyError):
        dg.node(reuse=False)(Sub)
        dg.static_resolve(Sub)

    dg.reset(clear_nodes=True)

    dg.node(ignore=int)(Sub)
    dg.static_resolve(Sub)
    # assert dg.resolve(Sub, age=15).age == 15
    dg.resolve(Sub, age=15)


def test_empty_graph_merge():
    dg1 = DependencyGraph()
    dg2 = DependencyGraph()
    dg1.merge(dg2)


def test_empty_multiple_merge():
    dg1 = DependencyGraph()
    dg2 = DependencyGraph()
    dg3 = DependencyGraph()
    dg1.merge([dg2, dg3])


def test_graph_merge_with_error():
    dg = DependencyGraph()

    dg2 = DependencyGraph()

    with pytest.raises(MergeWithScopeStartedError):
        with dg2.scope():
            dg.merge(dg2)


def test_graph_static_resolved_should_override():

    from .test_data import ComplianceChecker, DatabaseConfig

    dg = DependencyGraph()
    dg2 = DependencyGraph()

    dg.static_resolve(ComplianceChecker)
    c = dg.resolve(ComplianceChecker)

    dg2.static_resolve(DatabaseConfig)
    d = dg2.resolve(DatabaseConfig)
    repr(dg.nodes[ComplianceChecker].config)

    dg.merge(dg2)
    assert ComplianceChecker in dg and DatabaseConfig in dg

    c1, d1 = dg.resolve(ComplianceChecker), dg.resolve(DatabaseConfig)
    assert c1 is c
    assert d1 is d


def test_graph_static_resolved_should_not_override():

    from .test_data import ComplianceChecker, DatabaseConfig

    dg = DependencyGraph()
    dg2 = DependencyGraph()

    def checker_factory() -> ComplianceChecker: ...

    dg.node(checker_factory)

    dg2.static_resolve(ComplianceChecker)
    dg2.static_resolve(DatabaseConfig)

    dg.merge(dg2)
    assert dg.nodes[ComplianceChecker].factory_type == "function"
    assert ComplianceChecker in dg and DatabaseConfig in dg
