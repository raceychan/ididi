import typing as ty
from time import perf_counter

import pytest

from ididi import DependencyGraph

from .test_data import CLASSES

ROUNDS = 10


@pytest.fixture
def dependents() -> ty.Sequence[type]:
    # interesting
    # deps = CLASSES.values() would let dg
    # be substantially (3times) faster than [d for d in  ClASSES.values()]
    # return CLASSES.values()

    deps = [d for d in CLASSES.values()][:100]
    return deps


@pytest.fixture(scope="module")
def dg() -> DependencyGraph:
    return DependencyGraph()


@pytest.mark.benchmark
def test_register(dg: DependencyGraph, dependents: ty.Sequence[type]):
    pre = perf_counter()
    for dep in dependents:
        dg.node(dep)
    aft = perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to register {len(dependents)} classes")


@pytest.mark.benchmark
def test_static_resolve(dg: DependencyGraph):
    pre = perf_counter()
    dg.static_resolve_all()
    aft = perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to statically resolve {len(dg.nodes)} classes")


@pytest.mark.benchmark
def test_resolve_instances(dg: DependencyGraph, dependents: list[type]):
    total = 0

    for _ in range(ROUNDS):
        dg.reset()

        pre = perf_counter()
        for c in dependents:
            _ = dg.resolve(c)
        aft = perf_counter()
        cost = round(aft - pre, 6)
        total += cost

    avg = round(total / ROUNDS, 6)

    print(f"\n{avg} seoncds to resolve {len(dependents)} instances")


"""
0.012446 seoncds to register 122 classes
0.010367 seoncds to statically resolve 122 classes
0.001249 seoncds to resolve 122 instances
"""


"""
1.0.10
0.019022 seoncds to register 122 classes
0.00338 seoncds to statically resolve 122 classes
0.001264 seoncds to resolve 122 instances
"""

"""
1.1.0

0.020025 seoncds to register 122 classes
0.003467 seoncds to statically resolve 122 classes
0.001032 seoncds to resolve 122 instances
"""
