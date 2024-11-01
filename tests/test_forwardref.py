from ididi.graph import DependencyGraph

dag = DependencyGraph()


@dag.node
class ServiceA:
    def __init__(self, b: "ServiceB"):  # Forward reference to ServiceB
        self.b = b


@dag.node
class ServiceB:
    def __init__(self, c: "ServiceC"):
        self.c = c


@dag.node
class ServiceC:
    def __init__(self):
        pass


def test_forward_reference():
    # Resolve ServiceA which has a forward reference to ServiceB
    instance = dag.resolve(ServiceA)

    # Assert that the forward reference was properly resolved
    assert isinstance(instance, ServiceA)
    assert isinstance(instance.b, ServiceB)
    assert isinstance(instance.b.c, ServiceC)


# @pytest.mark.skip