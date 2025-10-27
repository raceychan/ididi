from dataclasses import dataclass
from time import perf_counter

import pytest

from ididi import Graph, use


@dataclass
class Config:
    name: str = "name"


# Factory functions for Level 4 (Leaf) Services
def create_l4_service1():
    return L4Service1(config=Config("l4_s1"))


def create_l4_service2():
    return L4Service2(config=Config("l4_s2"))


def create_l4_service3():
    return L4Service3(config=Config("l4_s3"))


# Level 4 Classes
class L4Service1:
    def __init__(self, config: Config):
        self.config = config


class L4Service2:
    def __init__(self, config: Config):
        self.config = config


class L4Service3:
    def __init__(self, config: Config):
        self.config = config


# Factory functions for Level 3
def create_l3_service1():
    return L3Service1(
        s1=create_l4_service1(), s2=create_l4_service2(), s3=create_l4_service3()
    )


def create_l3_service2():
    return L3Service2(
        s1=create_l4_service1(), s2=create_l4_service2(), s3=create_l4_service3()
    )


def create_l3_service3():
    return L3Service3(
        s1=create_l4_service1(), s2=create_l4_service2(), s3=create_l4_service3()
    )


# Level 3 Classes
class L3Service1:
    def __init__(self, s1: L4Service1, s2: L4Service2, s3: L4Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L3Service2:
    def __init__(self, s1: L4Service1, s2: L4Service2, s3: L4Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L3Service3:
    def __init__(self, s1: L4Service1, s2: L4Service2, s3: L4Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


# Factory functions for Level 2
def create_l2_service1():
    return L2Service1(
        s1=create_l3_service1(), s2=create_l3_service2(), s3=create_l3_service3()
    )


def create_l2_service2():
    return L2Service2(
        s1=create_l3_service1(), s2=create_l3_service2(), s3=create_l3_service3()
    )


def create_l2_service3():
    return L2Service3(
        s1=create_l3_service1(), s2=create_l3_service2(), s3=create_l3_service3()
    )


# Level 2 Classes
class L2Service1:
    def __init__(self, s1: L3Service1, s2: L3Service2, s3: L3Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L2Service2:
    def __init__(self, s1: L3Service1, s2: L3Service2, s3: L3Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L2Service3:
    def __init__(self, s1: L3Service1, s2: L3Service2, s3: L3Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


# Factory functions for Level 1
def create_l1_service1():
    return L1Service1(
        s1=create_l2_service1(), s2=create_l2_service2(), s3=create_l2_service3()
    )


def create_l1_service2():
    return L1Service2(
        s1=create_l2_service1(), s2=create_l2_service2(), s3=create_l2_service3()
    )


def create_l1_service3():
    return L1Service3(
        s1=create_l2_service1(), s2=create_l2_service2(), s3=create_l2_service3()
    )


# Level 1 Classes
class L1Service1:
    def __init__(self, s1: L2Service1, s2: L2Service2, s3: L2Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L1Service2:
    def __init__(self, s1: L2Service1, s2: L2Service2, s3: L2Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


class L1Service3:
    def __init__(self, s1: L2Service1, s2: L2Service2, s3: L2Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


# Root Service and its factory
class RootService:
    def __init__(self, s1: L1Service1, s2: L1Service2, s3: L1Service3):
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3


def create_root_service():
    return RootService(
        s1=create_l1_service1(), s2=create_l1_service2(), s3=create_l1_service3()
    )


async def create_root_service_reuse():
    # Level 4 (leaves)
    l4_s1 = L4Service1(config=Config("l4_s1"))
    l4_s2 = L4Service2(config=Config("l4_s2"))
    l4_s3 = L4Service3(config=Config("l4_s3"))

    # Level 3
    l3_s1 = L3Service1(s1=l4_s1, s2=l4_s2, s3=l4_s3)
    l3_s2 = L3Service2(s1=l4_s1, s2=l4_s2, s3=l4_s3)
    l3_s3 = L3Service3(s1=l4_s1, s2=l4_s2, s3=l4_s3)

    # Level 2
    l2_s1 = L2Service1(s1=l3_s1, s2=l3_s2, s3=l3_s3)
    l2_s2 = L2Service2(l3_s1, l3_s2, l3_s3)
    l2_s3 = L2Service3(l3_s1, l3_s2, l3_s3)

    # Level 1
    l1_s1 = L1Service1(l2_s1, l2_s2, l2_s3)
    l1_s2 = L1Service2(l2_s1, l2_s2, l2_s3)
    l1_s3 = L1Service3(l2_s1, l2_s2, l2_s3)

    # Root
    root = RootService(l1_s1, l1_s2, l1_s3)
    return root


# Usage


@pytest.mark.benchmark
async def test_create_root():
    n = 1000
    start = perf_counter()

    for _ in range(n):
        _ = create_root_service()
    end = perf_counter()

    menual = round((end - start), 6)

    print(f"\nmenaul construction took {menual}")

    dg = Graph()
    classes: dict[str, type] = {
        k: v
        for k, v in globals().items()
        if not k.startswith("@pytest") and isinstance(v, type) and v is not Graph
    }

    for c in classes.values():
        dg.node(c)

    start = perf_counter()
    for _ in range(n):
        dg.resolve(RootService)
    end = perf_counter()
    res = round((end - start), 6)

    print(f"ididi resolve took {res}")

    print(
        f"current implementation(without reuse) is {round(res / menual, 6)} times slower"
    )


@pytest.mark.benchmark
async def test_create_root_reuse():
    n = 1000
    start = perf_counter()
    for _ in range(n):
        _ = await create_root_service_reuse()
    end = perf_counter()
    menual = round(end - start, 6)

    print(f"menaul construction {menual}")

    dg = Graph()
    classes: dict[str, type] = {
        k: v
        for k, v in globals().items()
        if not k.startswith("@pytest") and isinstance(v, type) and v is not Graph
    }

    for c in classes.values():
        dg.node(use(c, reuse=True))

    dg.remove_dependent(RootService)
    dg.node(RootService)
    assert dg.nodes[RootService].reuse is False

    start = perf_counter()
    for _ in range(n):
        dg.resolve(RootService)
    end = perf_counter()
    res = round(end - start, 6)
    print(f"ididi resolve {res}")

    print(
        f"current implementation is {round(res / menual, 6)} times slower"
    )
