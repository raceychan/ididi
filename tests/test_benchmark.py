import typing as ty
from contextlib import contextmanager
from time import perf_counter

import pytest

from ididi import Graph, Ignore, Resource, use

from .test_data import CLASSES, UserService, user_service_factory

ROUNDS = 1000


@pytest.fixture
def dependents() -> ty.Sequence[type]:
    deps = [d for d in CLASSES.values()][:100]
    return deps


@pytest.fixture(scope="module")
def dg() -> Graph:
    return Graph()


@pytest.mark.benchmark
def test_register(dg: Graph, dependents: ty.Sequence[type]):
    pre = perf_counter()
    for dep in dependents:
        dg.node(dep)
    aft = perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to register {len(dependents)} classes")


@pytest.mark.benchmark
def test_analyze_nodes(dg: Graph):
    pre = perf_counter()
    dg.analyze_nodes()
    aft = perf_counter()
    cost = round(aft - pre, 6)
    print(f"\n{cost} seoncds to statically resolve {len(dg.nodes)} classes")


@pytest.mark.benchmark
def test_resolve_instances(dg: Graph, dependents: list[type]):
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
def test_entry(dg: Graph, dependents: list[type]):
    rounds = ROUNDS * 10
    dg.reset(clear_nodes=True)

    SERVICE_REGISTRY: set[UserService] = set()

    class Conn: ...

    class Repo: ...

    class Cache: ...

    @contextmanager
    def get_conn() -> Resource[Conn]:
        conn = Conn()
        yield conn

    @contextmanager
    def get_repo() -> Resource[Repo]:
        r = Repo()
        yield r

    @contextmanager
    def get_cache() -> Resource[Cache]:
        r = Cache()
        yield r

    def create_user(
        user_name: Ignore[str],
        user_email: Ignore[str],
        service: ty.Annotated[UserService, use(UserService)],
        conn: ty.Annotated[Conn, use(get_conn)],
        repo: ty.Annotated[Repo, use(get_repo)],
        cache: ty.Annotated[Cache, use(get_cache)],
    ) -> str:
        SERVICE_REGISTRY.add(service)
        return "ok"

    normal_total = 0

    for _ in range(rounds):
        pre = perf_counter()
        user_service = user_service_factory()
        with get_conn() as conn:
            with get_repo() as repo:
                with get_cache() as cache:
                    assert (
                        create_user(
                            "test",
                            "email",
                            service=user_service,
                            conn=conn,
                            repo=repo,
                            cache=cache,
                        )
                        == "ok"
                    )
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
    assert dg.nodes[UserService].reuse is False

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

    print(f"current implementation of entry is {ratio} times slower")


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

"""
1.3.3

tests/test_benchmark.py 
0.015494 seoncds to register 100 classes
0.002466 seoncds to statically resolve 100 classes
0.000141 seoncds to resolve 100 instances

0.001677 seoncds to call regular function create_user 1000 times
0.011296 seoncds to call entry version of create_user 1000 times
current implementation of entry is 6.735838 times slower
current implementation(without reuse) is 8.631982 times slower
"""

"""
1.3.4

0.02155 seoncds to register 100 classes
0.003622 seoncds to statically resolve 100 classes
0.000232 seoncds to resolve 100 instances

0.002916 seoncds to call regular function create_user 1000 times
0.018271 seoncds to call entry version of create_user 1000 times

current implementation of entry is 6.265775 times slower


menaul construction 0.038749
ididi resolve 0.285463
current implementation(without reuse) is 7.366977 times slower
"""


"""
1.4.3

0.00866 seoncds to register 100 classes
0.002322 seoncds to statically resolve 100 classes
0.00012 seoncds to resolve 100 instances

0.001616 seoncds to call regular function create_user 1000 times
0.007332 seoncds to call entry version of create_user 1000 times

current implementation of entry is 4.537129 times slower

menaul construction took 0.024939
ididi resolve took 0.102979
current implementation(without reuse) is 4.129235 times slower
"""


"""
1.4.4

NOTE:
benchmark of entry after 1.4.4 is slower because we changed the test method of entry.

|─ 0.239 Graph.resolve  ididi/resolver.py:447
│  ├─ 0.167 resolve_dfs  ididi/resolver.py:79
│  │  ├─ 0.087 [self]  ididi/resolver.py
│  │  ├─ 0.031 Graph.resolve_callback  ididi/resolver.py:402
│  │  │  ├─ 0.017 register_dependent  ididi/resolver.py:64
│  │  │  └─ 0.014 [self]  ididi/resolver.py
│  │  ├─ 0.020 resolve_dfs  ididi/resolver.py:79
│  │  │  ├─ 0.013 [self]  ididi/resolver.py
│  │  │  └─ 0.007 dict.get  <built-in>
│  │  ├─ 0.018 Dependencies.__iter__  ididi/_node.py:204
│  │  └─ 0.007 dict.get  <built-in>
│  ├─ 0.048 [self]  ididi/resolver.py
│  ├─ 0.017 NodeConfig.analyze  ididi/resolver.py:274
│  └─ 0.007 dict.get  <built-in>
└─ 0.016 [self]  tests/test_benchmark.py
"""

"""
1.4.5

menaul construction took 0.026685
ididi resolve took 0.027252
current implementation(without reuse) is 1.021248 times slower
"""
