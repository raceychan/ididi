from typing import Annotated

import pytest

from ididi import Graph, Ignore, use


def test_complex_resolve():
    dg = Graph()

    def dep3(e: int) -> Ignore[str]:
        return "ok"

    def dependency(a: int, a2: Annotated[str, use(dep3)]) -> Ignore[int]:
        return a

    def dep2(k: str, g: Annotated[int, use(dependency)]) -> Ignore[str]:
        assert g == 1
        return k

    def main(
        a: int,
        b: int,
        c: Annotated[int, use(dependency)],
        d: Annotated[str, use(dep2)],
    ) -> Ignore[float]:
        assert d == "f"
        return a + b + c

    res = dg.resolve(main, a=1, b=2, k="f", e="e")
    assert res == 4


def test_complex_resolve_fail():
    dg = Graph()

    def dep3(e: int) -> Ignore[str]:
        return "ok"

    def dependency(a: int, a2: Annotated[str, use(dep3)]) -> Ignore[int]:
        return a

    def dep2(k: str, g: Annotated[int, use(dependency)]) -> Ignore[str]:
        assert g == 1
        return k

    def main(
        a: int,
        b: int,
        c: Annotated[int, use(dependency)],
        d: Annotated[str, use(dep2)],
    ) -> Ignore[float]:
        assert d == "f"
        return a + b + c

    with pytest.raises(TypeError):
        dg.resolve(main, a=1, b=2, k="f")
