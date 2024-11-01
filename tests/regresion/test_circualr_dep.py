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


dag.reset()


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


@pytest.mark.skip(reason="not implemented")
def test_advanced_cycle_detection():
    """
    DependentNode.resolve_forward_dependency

    def resolve_forward_dependency(self, dep: ForwardDependent) -> type:
        Resolve a forward dependency and check for circular dependencies.
        Returns the resolved type and its node.

        I would argue that whenever there is a circular dependency,
        it would start with a forward reference.

        we might improve circular dependencies by carrying a stack of nodes
        A -> B -> C -> D -> A
        we carry [A, B, C, D] when we detect circular dependency
        resolve_type_stack = []
        resolved_type = dep.resolve()
        resolve_dep_stack.append(resolved_type)
        for sub_param in get_full_typed_signature(resolved_type).parameters.values():
            if sub_param.annotation in resolve_dep_stack:
                raise CircularDependencyDetectedError(resolve_dep_stack)
    """

    with pytest.raises(CircularDependencyDetectedError) as exc_info:
        dag.resolve(A)

    assert exc_info.value.cycle_path == [A, B, C, D]
