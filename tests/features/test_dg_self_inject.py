from ididi import DependencyGraph


class UserService: ...


def get_service(dg: DependencyGraph) -> UserService:
    return dg


def test_dg_self_inject():
    dg = DependencyGraph()
    assert dg is dg.resolve(get_service)
