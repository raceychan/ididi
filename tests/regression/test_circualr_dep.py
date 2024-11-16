import pytest

from ididi import DependencyGraph
from ididi.errors import CircularDependencyDetectedError


class CircleServiceA:
    def __init__(self, b: "CircleServiceB"):
        self.b = b


class CircleServiceB:
    def __init__(self, a: CircleServiceA):
        self.a = a


def test_cycle_detection():
    dag = DependencyGraph()
    with pytest.raises(CircularDependencyDetectedError) as e:
        dag.resolve(CircleServiceA)

    assert e.value.cycle_path == [CircleServiceA, CircleServiceB, CircleServiceA]


class A:
    def __init__(self, b: "B"):
        self.b = b


class B:
    def __init__(self, a: "C"):
        self.a = a


class C:
    def __init__(self, d: "D"):
        pass


class D:
    def __init__(self, a: A):
        self.a = a


def test_advanced_cycle_detection():
    """
    DependentNode.resolve_forward_dependency
    """
    dag = DependencyGraph()

    with pytest.raises(CircularDependencyDetectedError) as exc_info:
        dag.static_resolve(A)

    assert exc_info.value.cycle_path == [A, B, C, D, A]
