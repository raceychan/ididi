from concurrent.futures import ThreadPoolExecutor

import pytest

from ididi import DependencyGraph


class Config:
    def __init__(self, file: str = "file"):
        self.file = file


class Database:
    def __init__(self, config: Config):
        self.config = config


class Repository:
    def __init__(self, db: Database):
        self.db = db


class AuthService:
    def __init__(self, repo: Repository):
        self.repo = repo


@pytest.fixture(scope="function")
def dg():
    return DependencyGraph()


@pytest.fixture
def max_workers():
    return 10000


@pytest.fixture
def pool(max_workers: int):
    return ThreadPoolExecutor(max_workers)


def test_repeat_resolve(dg: DependencyGraph):
    instances: list[object] = []

    dg.node(reuse=False)(AuthService)
    dg.static_resolve(AuthService)

    def resolve(dg: DependencyGraph):
        obj = dg.resolve(AuthService)
        instances.append(id(obj))
        return obj

    res = [resolve(dg) for _ in range(1000)]

    assert len(set(res)) == len(res)


def test_threading_resolve_non_reuse(
    dg: DependencyGraph, pool: ThreadPoolExecutor, max_workers: int
):

    # Create a thread pool executor
    results: list[object] = []

    dg.node(reuse=False)(AuthService)
    dg.static_resolve(AuthService)

    def resolve(dg: DependencyGraph):
        obj = dg.resolve(AuthService)
        results.append(obj)
        return id(obj)

    with pool as executor:
        # Submit multiple resolve tasks
        futures = [executor.submit(resolve, dg) for _ in range(max_workers)]
        for future in futures:
            future.result()  # Collect results

    assert len(set(results)) == len(results)


def test_threading_resolve_reuse(
    dg: DependencyGraph, pool: ThreadPoolExecutor, max_workers: int
):

    # Create a thread pool executor
    results: list[object] = []

    dg.node(reuse=True)(AuthService)

    def resolve(dg: DependencyGraph):
        return id(dg.resolve(AuthService))

    with pool as executor:
        # Submit multiple resolve tasks
        futures = [executor.submit(resolve, dg) for _ in range(max_workers)]
        for future in futures:
            results.append(future.result())  # Collect results

    assert len(set(results)) == 1
