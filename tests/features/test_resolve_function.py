from typing import Annotated

from ididi import Graph, Ignore, Resource, use

from ..test_data import Config, UserService


class User:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role


class Connection:
    def __init__(self, config: Config):
        self.config = config


def get_conn(config: Config) -> Resource[Connection]:
    conn = Connection(config)
    yield conn


def get_user(
    conn: Annotated[Connection, use(get_conn)], config: Config
) -> Ignore[User]:
    assert isinstance(conn, Connection)
    assert isinstance(config, Config)
    return User("user", "admin")


def validate_admin(
    user: Annotated[User, use(get_user)], service: UserService
) -> Ignore[str]:
    assert user.role == "admin"
    assert isinstance(service, UserService)
    return "ok"


class Route:
    def __init__(self, validte_permission: Annotated[str, use(validate_admin)]):
        assert validte_permission == "ok"


async def test_resolve_function():
    dg = Graph()

    with dg.scope() as scope:
        assert isinstance(scope.resolve(get_user), User)


def test_dg_resolve_params():
    dg = Graph()

    with dg.scope() as scope:
        assert scope.resolve(validate_admin) == "ok"


def test_dg_resolve_cls_depends_on_function():
    dg = Graph()
    with dg.scope() as scope:
        assert isinstance(scope.resolve(Route), Route)

    assert (node := dg.get(validate_admin)) and node.dependent is validate_admin
