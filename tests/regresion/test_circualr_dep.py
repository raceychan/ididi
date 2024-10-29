import pytest

from ididi import DependencyGraph
from ididi.errors import CircularDependencyDetectedError

dag = DependencyGraph()


@dag.node
class CircleServiceA:
    def __init__(self, b: "CircleServiceB"):
        self.b = b


@dag.node
class CircleServiceB:
    def __init__(self, a: CircleServiceA):
        self.a = a


def test_cycle_detection():
    with pytest.raises(CircularDependencyDetectedError) as exc_info:
        dag.resolve(CircleServiceA)

    assert exc_info.value.cycle_path == [CircleServiceA, CircleServiceB]
