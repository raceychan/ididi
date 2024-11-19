import time

import pytest


@pytest.mark.benchmark
def test_static_resolve():

    from ididi import DependencyGraph

    from .test_data import CLASSES

    dg = DependencyGraph()
    pre = time.perf_counter()
    for k, c in CLASSES.items():
        dg.node(c)
    aft = time.perf_counter()
    cost = round(aft - pre, 6)
    print(f"{cost} seoncds to register {len(CLASSES)} classes")

    pre = time.perf_counter()
    dg.static_resolve_all()
    aft = time.perf_counter()

    cost = round(aft - pre, 6)
    print(f"{cost} seoncds to statically resolve {len(dg.nodes)} classes")

    t = CLASSES["ConflictResolver"]
    pre = time.perf_counter()
    r = dg.resolve(t)
    aft = time.perf_counter()

    cost = round(aft - pre, 6)

    deps = dg.visitor.get_dependencies(t, recursive=True)
    print(f"{cost} seoncds to resolve {t} with {len(deps)} dependencies")

    # sorted_deps = dg.visitor.top_sorted_dependencies()
