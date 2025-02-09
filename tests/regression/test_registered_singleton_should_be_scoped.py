from ididi import Graph


class Time:
    def __init__(self, name: str):
        self.name = name


def test_registered_singleton():
    """
    previously
    dg.should_be_scoped(dep_type) would analyze dependencies of the
    dep_type, and if it contains resolvable dependencies exception would be raised

    since registered_singleton is managed by user,
    dg.should_be_scoped(registered_singleton) should always be False.
    """
    timer = Time("1")

    dg = Graph()

    dg.register_singleton(timer)

    # this would raise error in 1.2.0
    assert dg.should_be_scoped(Time) is False
