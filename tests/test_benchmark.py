import typing as ty
from time import perf_counter

import pytest

from ididi import DependencyGraph

from .test_data import CLASSES, UserService, user_service_factory

ROUNDS = 1000


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


@pytest.mark.benchmark
def test_entry(dg: DependencyGraph, dependents: list[type]):
    rounds = ROUNDS * 100
    dg.reset()

    def create_user(user_name: str, user_email: str, service: UserService):
        return "ok"

    total = 0

    for _ in range(rounds):
        pre = perf_counter()
        user_service = user_service_factory()
        assert create_user("test", "email", service=user_service) == "ok"
        aft = perf_counter()
        cost = round(aft - pre, 12)
        total += cost

    print(
        f"\n{round(total, 6)} seoncds to call regular function {create_user.__name__} {rounds} times"
    )

    create_user = dg.entry(create_user)

    total = 0
    for _ in range(rounds):
        pre = perf_counter()
        assert create_user("test", "email") == "ok"
        aft = perf_counter()
        cost = round(aft - pre, 6)
        total += cost

    print(
        f"\n{round(total, 6)} seoncds to call entry version of {create_user.__name__} {rounds} times"
    )


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

"""
1.1.2
0.010674 seoncds to register 100 classes
0.011788 seoncds to statically resolve 100 classes
0.000933 seoncds to resolve 100 instances
0.089221 seoncds to call <function test_entry.<locals>.create_user at 0x7f377a3a8820> 10000
0.412887 seoncds to call <function test_entry.<locals>.create_user at 0x7f377a3a8940>
"""

"""
1.1.3

0.018514 seoncds to register 100 classes
0.003045 seoncds to statically resolve 100 classes
0.000878 seoncds to resolve 100 instances
0.099846 seoncds to call regular function create_user 100000 times
0.104534 seoncds to call entry version of create_user 100000 times
"""
