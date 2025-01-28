from ididi import DependencyGraph, Ignore, use

from ..test_data import AuthenticationService, Database, DatabaseConfig, UserService


def test_ignore_param():
    class User:
        def __init__(self, name: Ignore[str]):
            self.name = name

    dg = DependencyGraph()

    dg.analyze(User)
    node = dg.nodes[User]
    assert node.config.ignore

    u = dg.resolve(User, name="test")

    assert u.name == "test"


def test_resolve():
    dg = DependencyGraph()

    def db_fact(dg: DependencyGraph) -> Database:
        return Database(config=dg.resolve(DatabaseConfig))

    def service_factory(
        *, db: Database = use(db_fact), auth: AuthenticationService, name: str
    ) -> UserService:
        return UserService(db=db, auth=auth)

    dg.resolve(service_factory, name="aloha")


def test_graph_ignore_name():
    dg = DependencyGraph(ignore="name")

    def create_db(name: str) -> Database:
        return Database(1)

    dg.analyze(create_db)
    assert isinstance(dg.resolve(create_db, name=5), Database)


def test_graph_ignore_many_names():
    from datetime import datetime

    dg = DependencyGraph(ignore=("name", "age", datetime))

    def create_db(name: str, age: int, dt: datetime) -> Database:
        return Database(1)

    dg.analyze(create_db)


def test_node_with_many_ignore():
    dg = DependencyGraph()

    dg.node(ignore="config")(Database)