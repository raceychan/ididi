from ididi.node import DependentNode


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


# @pytest.fixture
# def basic_nodes():
#     return {
#         "A": DependentNode.from_node(A),
#         "B": DependentNode.from_node(B),
#         "Config": DependentNode.from_node(Config),
#     }


def test_complex_signature():
    node = DependentNode.from_node(ComplexDependency)
    node.dependent.dependent_name


# def test_factory_function():
#     factory_node = DependentNode.from_node(complex_factory)

#     factory_obj = factory_node.build()

#     assert isinstance(factory_obj, ComplexDependency)
#     assert isinstance(factory_obj.a, A)
#     assert factory_obj.a.a == 5
#     assert isinstance(factory_obj.b, B)
#     assert factory_obj.b.b == "b"
#     assert factory_obj.args == (1, 2, 3)
#     assert isinstance(factory_obj.config, Config)
#     assert factory_obj.config.env == "dev"
#     assert factory_obj.kwargs == {"extra": "test"}


def test_typed_annotation():
    # Test forward reference handling
    class ServiceA:
        def __init__(self, nums: list[int]):
            self.nums = nums

    DependentNode.from_node(ServiceA)


def test_empty_init():
    # Test class with no __init__
    class EmptyService:
        pass

    node = DependentNode.from_node(EmptyService)
    assert not node.signature


# def test_factory_without_return_type():
#     # Test factory without return type annotation
#     def bad_factory():
#         return object()

#     with pytest.raises(MissingReturnTypeError):
#         DependentNode.from_node(bad_factory)


# def test_node_without_annotation():

#     class Service:
#         def __init__(self, a):
#             self.a = a

#     node = DependentNode.from_node(Service)
#     with pytest.raises(UnsolvableDependencyError):
#         node.build()


# def test_weird_annotation():
#     class Weird:
#         def __init__(self, a: int | str):
#             self.a = a

#     f = DependentNode.from_node(Weird)
#     with pytest.raises(UnsolvableDependencyError):
#         f.build()

#     def weird_factory(a: int | str = 3) -> Weird:
#         return Weird(a)

#     f = DependentNode.from_node(weird_factory)
#     f.build()


def test_varidc_keyword_args():
    class Service:
        def __init__(self, **kwargs: int):
            self.kwargs = kwargs

    node = DependentNode.from_node(Service)
    assert node.signature.dprams["kwargs"].unresolvable
    repr(node)


# def test_node_repr():
#     node = DependentNode.from_node(Service)
#     assert "Service" in str(node)


# def test_node_without_params():
#     node = DependentNode.from_node(object)
#     assert not node.signature
#     assert "object" in str(node)
#     node.build()

#     with pytest.raises(UnsolvableDependencyError):
#         DependentNode.from_node(int).build()


# def test_forward_dependent_repr():
#     class A:
#         def __init__(self, b: "B"):
#             self.b = b

#     class B:
#         def __init__(self, a: int):
#             self.a = a

#     node = DependentNode.from_node(A)

#     str(node)
#     str(node.dependent)


# def test_complex_union_type():
#     class Service:
#         def __init__(self, union_builtins: dict[str, str] | list[int]):
#             self.union_builtins = union_builtins

#     node = DependentNode.from_node(Service)
#     with pytest.raises(UnsolvableDependencyError):
#         res = node.build()


# def test_complex_union_type_2():
#     class A:
#         def __init__(self, *args: int):
#             self.args = args

#     class B:
#         def __init__(self, *args: str):
#             self.args = args

#     class Service:
#         def __init__(self, *args: A | B):
#             self.args = args

#     node = DependentNode.from_node(Service)
#     for dep in node.signature:
#         repr(dep)

#     with pytest.raises(UnsolvableDependencyError):
#         res = node.build()


# def test_build_node_with_override():
#     class DataBase:
#         def __init__(self, engine: str = "mysql", /, driver: str = "aiomysql"):
#             self.engine = engine
#             self.driver = driver

#     class Repository:
#         def __init__(self, db: DataBase):
#             self.db = db

#     class UserService:
#         def __init__(self, repository: Repository):
#             self.repository = repository

#     node = DependentNode.from_node(UserService)
#     instance = node.build(repository=dict(db=dict(engine="sqlite")))
#     assert isinstance(instance, UserService)
#     assert isinstance(instance.repository, Repository)
#     assert isinstance(instance.repository.db, DataBase)
#     assert instance.repository.db.engine == "sqlite"
#     assert instance.repository.db.driver == "aiomysql"


# def test_node_repr(): ...
