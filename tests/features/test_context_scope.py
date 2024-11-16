import pytest

from ididi import DependencyGraph


@pytest.mark.asyncio
async def test_scope_different_across_context():
    dg = DependencyGraph()

    class Normal:
        def __init__(self, name: str = "normal"):
            self.name = name

    def func1():
        with dg.scope(1) as s1:
            repr(s1)
            return s1

    func1()
