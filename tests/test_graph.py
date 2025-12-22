import inspect
import sys
import typing as ty
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Annotated, Optional
from unittest import mock

import pytest

from ididi import AsyncResource, Graph, Ignore, Resolver, use
from ididi.errors import (
    ABCNotImplementedError,
    AsyncResourceInSyncError,
    ConfigConflictError,
    DeprecatedError,
    MergeWithScopeStartedError,
    MissingAnnotationError,
    NotSupportedError,
    ParamReusabilityConflictError,
    PositionalOverrideError,
    TopLevelBulitinTypeError,
)
from ididi.interfaces import MISSING

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
    return Graph()


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


def test_dag_resolve(dg: Graph):
    @dg.node
    class UserService:
        def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
            self.repo = repo
            self.auth = auth

    service = dg.resolve(UserService)
    assert isinstance(service.repo.db, Database)
    assert isinstance(service.repo.cache, Cache)
    assert isinstance(service.auth.db, Database)


def test_get_dependency_types(dg: Graph):
    database_dependencies = dg.visitor.get_dependencies(Database)
    assert len(database_dependencies) == 1
    assert Config in database_dependencies


def test_get_dependent_types(dg: Graph):
    database_dependents = dg.visitor.get_dependents(Database)
    assert len(database_dependents) == 2
    assert UserRepository in database_dependents
    assert AuthService in database_dependents


def test_node_signature_change_after_factory(dg: Graph):
    class UserService:
        def __init__(
            self,
            repo: UserRepository,
            auth: AuthService,
            name: str = "user",
        ):
            self.repo = repo
            self.auth = auth
            self.name = name

    dg.analyze(UserService)
    node = dg.nodes[UserService]
    print(node.dependencies)

    assert len(node.dependencies) == 3

    def user_service_factory(repo: UserRepository) -> UserService:
        return UserService(repo, auth=AuthService(Database(Config())))

    dg.node(user_service_factory)
    node = dg.nodes[UserService]
    assert len(node.dependencies) == 1


def test_top_level_builtin_dependency(dg: Graph):
    with pytest.raises(TopLevelBulitinTypeError):
        dg.resolve(int)


def test_missing_implementation(dg: Graph):
    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    with pytest.raises(ABCNotImplementedError):
        dg.resolve(Repository)


def test_multiple_implementations(dg: Graph):
    dg.reset(clear_nodes=True)

    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    class Repo1(Repository):
        def save(self) -> None:
            pass

    dg.node(Repo1)

    class Repo2(Repository):
        def save(self) -> None:
            pass

    dg.node(Repo2)

    dg.resolve(Repository)


def test_multiple_implementations_with_factory(dg: Graph):
    class Repository(ABC):
        def __init__(self):
            pass

        @abstractmethod
        def save(self) -> None:
            """Save the repository data."""
            pass

    class Repo1(Repository):
        def save(self) -> None:
            pass

    class Repo2(Repository):
        def save(self) -> None:
            pass

    dg.node(Repo2)

    def repo_factory() -> Repository:
        return Repo1()

    dg.node(repo_factory)

    assert Repository in dg.nodes

    repo = dg.resolve(Repository)
    assert isinstance(repo, Repo1)


def test_static_resolve_factory(dg: Graph):
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

    class Repo2(Repository):
        def save(self) -> None:
            pass

    dg.node(Repo2)

    @dg.node
    def repo_factory() -> Repository:
        return Repo1()

    assert Repository in dg.nodes

    _ = dg.analyze(repo_factory)
    assert Repository in dg.type_registry[Repository]


async def test_resource_cleanup(dg: Graph):
    closed_resources: set[str] = set()

    @dg.node
    class Resource1:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            await self.close()

        async def close(self):
            closed_resources.add("resource1")

    @dg.node
    class Resource2:
        def __init__(self, r1: Resource1):
            self.r1 = r1

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            await self.close()

        async def close(self):
            closed_resources.add("resource2")

    # Resolve and verify resources
    with pytest.raises(AsyncResourceInSyncError):
        with dg.scope() as scope:
            instance = scope.resolve(Resource2)

    async with dg.ascope() as scope:
        instance = await scope.resolve(Resource2)
        assert isinstance(instance, Resource2)

    assert closed_resources == {"resource1", "resource2"}


# def test_node_removal(dg: Graph):
#     @dg.node
#     class Service:
#         pass

#     node = dg.nodes[Service]
#     dg.remove_node(node)

#     assert Service not in dg.nodes

#     dg.resolve(Service)


def test_graph_reset(dg: Graph):
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


def test_node_removal_cleanup(dg: Graph):
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
    dg.remove_dependent(Dependency)

    # Check that all references are cleaned up
    assert Dependency not in dg.nodes
    assert Dependency not in dg.type_registry
    assert Dependency not in dg.resolution_registry
    assert Dependency not in dg.resolved_nodes


def test_facwry_override(dg: Graph):
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


def test_factory_registration(dg: Graph):
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


def test_unsupported_annotation(dg: Graph):
    @dg.node
    class BadService:
        def __init__(self, bad: None):  # object is not a proper annotation
            self.bad = bad

    with pytest.raises(TypeError):
        dg.resolve(BadService)

    class New:
        def __init__(self, name: str): ...

    dg.resolve(New, name="aloha")

    class Old:
        def __init__(self, age: int = 3): ...

    def new_factory(age: int) -> New:
        return New(age)

    dg.node(new_factory)

    dg.resolve(New, age=5)


def test_dependency_override(dg: Graph):
    @dg.node
    class Service:
        def __init__(self, name: str = "default"):
            self.name = name

    # Override with kwargs
    instance = dg.resolve(Service, name="overridden")
    assert instance.name == "overridden"


def test_nested_dependency_override(dg: Graph):
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


def test_abstract_base_resolution(dg: Graph):
    dg = Graph()

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


def test_multiple_dependency_paths(dg: Graph):
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

    dg.node(use(Shared, reuse=True))
    instance = dg.resolve(Root)
    assert dg.resolution_registry[Shared] is instance.s1.shared2
    assert dg.resolution_registry[Shared] is instance.s2.shared1
    # Verify shared instance is actually shared
    assert instance.s1.shared2 is instance.s2.shared1
    assert instance.s1.shared2.value == instance.s2.shared1.value == "shared"


def test_type_mapping_cleanup(dg: Graph):
    class Interface(ABC):
        pass

    @dg.node
    class Implementation(Interface):
        pass

    # Verify type mapping
    assert Implementation in dg.type_registry[Interface]

    # Remove implementation
    node = dg.nodes[Implementation]
    dg.reset(clear_nodes=True)

    # Verify type mapping is cleaned up
    assert Interface not in dg.type_registry or not dg.type_registry[Interface]


@pytest.mark.asyncio
async def test_graph_without_static_resolve(dg: Graph):
    # This test specifically needs a new dag instance
    dg = Graph()

    @dg.node
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


def test_remove_old_node():
    dg = Graph()

    class IUser: ...

    class User(IUser):
        def __init__(self, name: str = "user"):
            self.name = name

    dg.node(User)

    def user_factory() -> User:
        return User("hello")

    dg.node(user_factory)

    assert IUser in dg.type_registry
    assert User in dg.type_registry
    assert User in dg.type_registry[IUser]


def test_resolve_node_without_annotation():
    dag = Graph()

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


def test_generic_service_is_supported(dg: Graph):
    dg.resolve(GenericService[str], item=5)


def test_generic_service_with_default(dg: Graph):
    class GenericService(ty.Generic[T]):
        def __init__(self, item: T = "5"):
            self.item = item

    # this should fail, since it is confusing behavior
    dg.resolve(GenericService[str])

    repr(dg)


def test_node_config_non_transitive(dg: Graph):
    class Base:
        def __init__(self, name: str = "base"): ...

    class Sub:
        def __init__(self, b: Base): ...

    dg.node(use(Sub, reuse=True))
    dg.analyze(Sub)

    assert dg.nodes[Base].reuse == MISSING


def test_partial_node(dg: Graph):
    class Base:
        def __init__(self, name: str = "base"): ...

    class Sub:
        def __init__(self, b: Base, age: int):
            self.b = b
            self.age = age

    dg.node(Sub)
    dg.analyze(Sub)

    dg.reset(clear_nodes=True)

    dg.node(ignore=int)(Sub)
    dg.analyze(Sub)
    # assert dg.resolve(Sub, age=15).age == 15
    dg.resolve(Sub, age=15)


def test_empty_graph_merge():
    dg1 = Graph()
    dg2 = Graph()
    dg1.merge(dg2)


def test_empty_multiple_merge():
    dg1 = Graph()
    dg2 = Graph()
    dg3 = Graph()
    dg1.merge([dg2, dg3])


def test_graph_merge_with_error():
    dg = Graph()

    dg2 = Graph()

    with pytest.raises(MergeWithScopeStartedError):
        with dg2.scope():
            dg.merge(dg2)


def test_graph_static_resolved_should_override():

    from .test_data import ComplianceChecker, DatabaseConfig

    dg = Graph()
    dg2 = Graph()

    dg.node(use(ComplianceChecker, reuse=True))
    dg.analyze(ComplianceChecker)
    c = dg.resolve(ComplianceChecker)

    dg.node(use(DatabaseConfig, reuse=True))
    dg2.node(use(DatabaseConfig, reuse=True))
    dg2.analyze(DatabaseConfig)
    d = dg2.resolve(DatabaseConfig)
    repr(dg.nodes[ComplianceChecker].reuse)

    dg.merge(dg2)
    assert ComplianceChecker in dg and DatabaseConfig in dg

    c1, d1 = dg.resolve(ComplianceChecker), dg.resolve(DatabaseConfig)
    assert c1 is c
    assert d1 is d


def test_graph_static_resolved_should_not_override():

    from .test_data import ComplianceChecker, DatabaseConfig

    dg = Graph()
    dg2 = Graph()

    def checker_factory() -> ComplianceChecker: ...

    dg.node(checker_factory)

    dg2.analyze(ComplianceChecker)
    dg2.analyze(DatabaseConfig)

    dg.merge(dg2)
    assert dg.nodes[ComplianceChecker].factory_type == "function"
    assert ComplianceChecker in dg and DatabaseConfig in dg


def test_factory_override_default_value():
    dg = Graph()

    class Database: ...

    DB_SINGLETON = Database()

    class UserRepos:
        def __init__(self, db: Optional[Database] = None):
            self.db = db

    class Person:
        def __init__(self, name: str = "p"):
            self.name = name

    def db_factory() -> Database:
        return DB_SINGLETON

    db = dg.resolve(UserRepos).db
    assert isinstance(db, Database)
    assert db is not DB_SINGLETON
    dg.node(db_factory)
    assert dg.resolve(Database) is DB_SINGLETON

    assert dg.resolve(Person).name == "p"


def test_graph_ignore():
    class Timer:
        def __init__(self, d: datetime):
            self._d = d

    dg = Graph(ignore=datetime)
    with pytest.raises(TypeError):
        dg.resolve(Timer)


def test_remove_dependent():
    class Database: ...

    dg = Graph()

    dg.remove_dependent(Database)


def test_remove_dependent_with_factory():
    dg = Graph()

    class Service: ...

    def sfactory() -> Service: ...

    dg.node(sfactory)

    dg.remove_dependent(sfactory)

    assert Service not in dg


def test_remove_new_type():
    dg = Graph()

    UserId = ty.NewType("UserId", str)

    def user_id_factory() -> UserId:
        return UserId("1")

    dg.node(user_id_factory)

    node = dg.search("UserId")

    assert node

    dg.remove_dependent(UserId)
    assert dg.search("UserId") is None


def test_graph_analyze_nested_annt():
    dg = Graph()

    class User: ...

    def hello() -> User: ...

    def build_user() -> ty.Annotated[User, use(hello)]:
        return User()

    dg.analyze(build_user)

    class Aloha: ...

    annt = ty.Annotated[Aloha, ty.Annotated[User, "hello"]]

    dg.analyze(annt)
    assert Aloha in dg


class Book:
    def __init__(self, b: str):
        self.b = b

    @classmethod
    def from_article(cls, a: str) -> "Book":
        return Book(b=a)

    def dump(self) -> "Book":
        return self


def test_analyze_classmethod():
    dg = Graph()

    dg.node(Book.from_article)

    node = dg.nodes[Book]
    assert node.factory_type == "function"
    assert node.factory.__func__ is Book.from_article.__func__  # type: ignore
    assert node.dependencies and any(param.name == "a" for param in node.dependencies)


def test_resolve_classmethod():
    dg = Graph()

    b = dg.resolve(Book.from_article, a="5")
    assert isinstance(b, Book)


def test_resolve_instance_method_raise_error():
    dg = Graph()

    with pytest.raises(NotSupportedError):
        dg.resolve(Book("b").dump)


class Conn:
    def __init__(self, url: str): 
        self.url = url


async def get_conn() -> Conn:
    return Conn("")


async def test_aresolve_override():
    dg = Graph()

    c = await dg.aresolve(Conn, url="url")
    assert c.url == "url"


def test_dg_get_node():
    dg = Graph()

    dg.node(get_conn)
    assert (node := dg.get(get_conn))
    assert node.dependent is Conn

    dg._nodes = 5


def test_factory_return_annt():
    dg = Graph()

    def get_conn() -> ty.Annotated[Conn, "aloha"]:
        return Conn("ahh")

    class DB:
        def __init__(self, conn: ty.Annotated[Conn, use(get_conn)]): ...

    dg.analyze(DB)


async def test_ididi_cheatsheet():
    class Base:
        def __init__(self, source: str = "class"):
            self.source = source

    class CTX(Base):
        def __init__(self, source: str = "class"):
            super().__init__(source)
            self.status = "init"

        async def __aenter__(self):
            self.status = "started"
            return self

        async def __aexit__(self, *args):
            self.status = "closed"

    class Engine(Base): ...

    class Connection(CTX):
        def __init__(self, engine: Engine):
            super().__init__()
            self.engine = engine

    def get_engine() -> Engine:
        return Engine("factory")

    side_effect: list[str] = []

    async def random_callback():
        nonlocal side_effect
        side_effect.append("callbacked")

    async def get_conn(engine: Engine) -> ty.AsyncGenerator[Connection, None]:
        async with Connection(engine) as conn:
            yield conn

    async def func_dep(engine: Engine, conn: Connection) -> Ignore[int]:
        return 69

    # ============= usage =============

    dg = Graph()
    assert isinstance(dg, Resolver)

    engine = dg.resolve(Engine)  # resolve a class
    assert isinstance(engine, Engine) and engine.source == "class"

    faq_engine = dg.resolve(get_engine)  # resolve a factory function of a class
    assert isinstance(faq_engine, Engine) and faq_engine.source == "factory"

    assert not side_effect

    async with dg.ascope() as ascope:
        ascope.register_exit_callback(random_callback)
        # register a callback to be called when scope is exited
        assert isinstance(ascope, Resolver)
        # NOTE: scopes are also resolvers, thus can have sub-scope
        conn = await ascope.aresolve(get_conn)
        # generator function will be transformed into context manager and can only be resolved within scope.
        assert isinstance(conn, Connection)
        assert conn.status == "started"
        # context manager is entered when scoped is entered.
        res = await ascope.aresolve(func_dep)
        assert res == 69
        # function dependencies are also supported

    assert conn.status == "closed"
    # context manager is exited when scope is exited.
    assert side_effect[0] == "callbacked"
    # registered callback will aslo be called.


async def test_graph_add_nodes():

    class Connection: ...

    async def get_conn() -> AsyncResource[Connection]:
        yield Connection()

    basic_graph = Graph()
    basic_graph.add_nodes(use(get_conn, reuse=True))


    main_graph = Graph()
    main_graph.merge(basic_graph)
    assert main_graph.nodes[Connection].reuse


def test_graph_add_nodes_tuple_deprecation_warning():
    graph = Graph()

    class Service: ...


    with pytest.raises(DeprecatedError):
        graph.add_nodes((Service, {"reuse": True, "ignore": [1, 2, 3]}))





async def test_graph_merge_with_node():
    g1 = Graph("g1")
    g2 = Graph("g2")
    
    class Resource: ...

    # @g2.node(reuse=True)
    @g1.node
    async def get_resource() -> ty.Annotated[AsyncResource[Resource], use(reuse=True)]:
        yield Resource()

    assert g1.nodes[Resource].reuse
        

async def test_dependent_conflicts():
    g = Graph("g1")

    
    @g.node(reuse=True)
    class Test:
        ...

    class B:
        def __init__(self, t: Test):
            self.t = t 

    class C:
        def __init__(self, t: Test):
            self.t = t
    
    if not g.nodes[Test].reuse:
        breakpoint()

    @g.node
    def get_c(t: Annotated[Test, use(reuse=True)]) -> C:
        return C(t)


    g.analyze_nodes()
    assert g.nodes[Test].reuse

    t1 = g.resolve(get_c).t
    t2 = g.resolve(get_c).t

    assert t1 is t2


def test_graph_analyze_reuse_dependent():
    dg = Graph()

    class User: ...

    def user_factory() -> Annotated[User, use(reuse=True)]:
        return User()

        
    assert dg.resolve(user_factory) is dg.resolve(user_factory)


def test_graph_analyze_reuse_dependentcy():
    dg = Graph()

    class User: ...

    @dg.node
    def user_factory() -> Annotated[User, use(reuse=True)]:
        return User()

    class UserManager:
        def __init__(self, user: User):
            self.user = user


    assert dg.nodes[User].reuse

    @dg.node
    def user_manager_factory(user: Annotated[User, use(user_factory, reuse=False)]) -> UserManager:
        return UserManager(user)


    with pytest.raises(ParamReusabilityConflictError):
        dg.analyze(user_manager_factory)

    with pytest.raises(ParamReusabilityConflictError):
        dg.resolve(user_manager_factory)


    # assert dg.nodes[User].config.ignore == (1,2,3)

    
def test_node_from_use():
    dg = Graph()

    
    class Service: ...

    dg.add_nodes(Annotated[Service, "hello"])
    assert dg.nodes[Service]


def test_ignore_param_and_should_be_ignored():
    dg = Graph()

    class User: ...

    @dg.node(ignore=("name", int))
    def test(name: Ignore[str], age: int) -> User:
        ...

    dg.should_be_scoped(test)



def test_param_error_message():
    
    dg = Graph()
    @dg.node
    class Test:
        ...

        
    class UserService: ...

    @dg.node
    def user_service(t: Annotated[Test, use(reuse=True)]) -> UserService:
        return UserService()


    dg.analyze(user_service)
