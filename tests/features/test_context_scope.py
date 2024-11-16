import pytest

from ididi import DependencyGraph

from ididi.errors import OutOfScopeError

@pytest.mark.asyncio
async def test_scope_different_across_context():
    dg = DependencyGraph()

    class Normal:
        def __init__(self, name: str = "normal"):
            self.name = name

    with dg.scope(1) as s1:

        def func1():
            with dg.scope(2) as s2:
                with dg.scope(3) as s3:
                    repr(s1)
                    s3.get_scope(1)
                    s3.get_scope(2)
                    s3.get_scope(3)
                    return s1

        func1()

        def func2():
            dg.use_scope(2)

        with pytest.raises(OutOfScopeError):
            func2()
