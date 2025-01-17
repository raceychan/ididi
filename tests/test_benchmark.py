import typing as ty
from time import perf_counter

import pytest

from ididi import DependencyGraph

from .test_data import CLASSES, UserService, user_service_factory

ROUNDS = 1000


@pytest.fixture
def dependents() -> ty.Sequence[type]:
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
    dg.analyze_nodes()
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
    rounds = ROUNDS * 1
    dg.reset(clear_nodes=True)

    SERVICE_REGISTRY: set[UserService] = set()

    def create_user(user_name: str, user_email: str, service: UserService) -> str:
        SERVICE_REGISTRY.add(service)
        return "ok"

    normal_total = 0

    for _ in range(rounds):
        pre = perf_counter()
        user_service = user_service_factory()
        assert create_user("test", "email", service=user_service) == "ok"
        aft = perf_counter()
        cost = round(aft - pre, 12)
        normal_total += cost

    assert len(SERVICE_REGISTRY) == rounds

    SERVICE_REGISTRY.clear()

    normal_total = round(normal_total, 6)

    print(
        f"\n{normal_total} seoncds to call regular function {create_user.__name__} {rounds} times"
    )

    create_user = dg.entry(reuse=False)(create_user)
    assert dg.nodes[UserService].config.reuse is False

    entry_total = 0
    for _ in range(rounds):
        pre = perf_counter()
        assert create_user("test", "email") == "ok"
        aft = perf_counter()
        cost = round(aft - pre, 6)
        entry_total += cost

    assert len(SERVICE_REGISTRY) == rounds

    entry_total = round(entry_total, 6)

    print(
        f"\n{entry_total} seoncds to call entry version of {create_user.__name__} {rounds} times"
    )

    ratio = round(entry_total / normal_total, 6)

    print(f"current implementation is {ratio} times slower")


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

"""
1.2.5

0.014147 seoncds to register 100 classes
0.002431 seoncds to statically resolve 100 classes
0.000558 seoncds to resolve 100 instances
0.001477 seoncds to call regular function create_user 1000 times
0.033853 seoncds to call entry version of create_user 1000 times

current implementation is 22.920108 times slower
"""

"""
1.2.7
0.015122 seoncds to register 100 classes
0.002498 seoncds to statically resolve 100 classes
0.000256 seoncds to resolve 100 instances
0.00165 seoncds to call regular function create_user 1000 times
0.021057 seoncds to call entry version of create_user 1000 times
current implementation is 12.761818 times slower
"""

"""
1.2.7
0.014272 seoncds to register 100 classes
0.002628 seoncds to statically resolve 100 classes
0.000139 seoncds to resolve 100 instances
0.001647 seoncds to call regular function create_user 1000 times
0.011167 seoncds to call entry version of create_user 1000 times
current implementation is 6.780206 times slower
"""
