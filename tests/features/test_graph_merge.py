from ididi import DependencyGraph
from tests.features.services import UserService


def service_factory() -> UserService:
    return UserService("random")


def test_graph_merge():
    dg1 = DependencyGraph()
    dg2 = DependencyGraph()

    dg1.node(service_factory)
    dg2.node(UserService)

    dg1.merge(dg2)

    assert dg1.nodes[UserService].factory_type == "function"

    service = dg1.resolve(UserService)

    assert service.repo == "random"
