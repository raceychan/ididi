from ididi import DependencyGraph, Ignore, use

from ..test_data import AuthenticationService, Database, DatabaseConfig, UserService


def test_ignore_param():
    class User:
        def __init__(self, name: Ignore[str]):
            self.name = name

    dg = DependencyGraph()

    dg.static_resolve(User)
    node = dg.nodes[User]
    assert node.config.ignore

    u = dg.resolve(User, name="test")

    assert u.name == "test"


def db_fact(dg: DependencyGraph) -> Database:
    return Database(config=dg.resolve(DatabaseConfig))


def service_factory(
    *, db: Database = use(db_fact), auth: AuthenticationService, name: Ignore[str]
) -> UserService:
    return UserService(db=db, auth=auth)


def test_resolve():
    dg = DependencyGraph()

    dg.resolve(service_factory, name="aloha")
