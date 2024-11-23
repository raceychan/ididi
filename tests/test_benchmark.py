import time
import typing as ty

import pytest

from ididi import DependencyGraph

from .test_data import CLASSES


@pytest.fixture
def dependents() -> ty.Sequence[type]:
    deps = CLASSES.values()
    return deps


@pytest.fixture(scope="module")
def dg() -> DependencyGraph:
    return DependencyGraph()


@pytest.mark.benchmark
def test_register(dg: DependencyGraph, dependents: ty.Sequence[type]):
    pre = time.perf_counter()
    for c in dependents:
        dg.node(c)
    aft = time.perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to register {len(dependents)} classes")


@pytest.mark.benchmark
def test_static_resolve(dg: DependencyGraph, dependents: ty.Sequence[type]):

    pre = time.perf_counter()
    dg.static_resolve_all()
    aft = time.perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to statically resolve {len(dg.nodes)} classes")


@pytest.mark.benchmark
def test_resolve_instances(dg: DependencyGraph, dependents: list[type]):

    pre = time.perf_counter()
    for c in dependents:
        _ = dg.resolve(c)
    aft = time.perf_counter()
    cost = round(aft - pre, 6)

    print(f"\n{cost} seoncds to resolve {len(dependents)} instances")


"""
0.012446 seoncds to register 122 classes
0.010367 seoncds to statically resolve 122 classes
0.001249 seoncds to resolve 122 instances
"""


