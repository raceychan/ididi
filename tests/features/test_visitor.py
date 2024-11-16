import pytest

from ididi import DependencyGraph


class B1: ...


class B2: ...


class B3: ...


class B:
    def __init__(self, b1: B1, b2: B2, b3: B3): ...


class C1: ...


class C2: ...


class C3: ...


class C:
    def __init__(self, c1: C1, c2: C2, c3: C3): ...


class D1: ...


class D2: ...


class D3: ...


class D:
    def __init__(self, d1: D1, d2: D2, d3: D3): ...


class A:
    def __init__(self, b: B, c: C, d: D): ...


@pytest.mark.debug
def test_visitor():
    dg = DependencyGraph()
    dg.static_resolve(A)

    assert len(dg.visitor.get_dependencies(A, recursive=False)) == 3
    assert len(dg.visitor.get_dependencies(A, recursive=True)) == 12

    assert len(dg.visitor.get_dependents(D1)) == 1


class ServiceA:
    def __init__(self, b: "ServiceB"):
        self.b = b


class ServiceC:
    pass


class ServiceB:
    def __init__(self, c: ServiceC):
        self.c = c


def test_initialization_order():
    dg = DependencyGraph()

    dg.static_resolve(ServiceA)
    order: list[type] = dg.visitor.top_sorted_dependencies()

    assert order.index(ServiceC) < order.index(ServiceB)
    assert order.index(ServiceB) < order.index(ServiceA)
