from contextlib import asynccontextmanager, contextmanager

from ididi._type_resolve import is_function


def test_function_check():
    def test_plain():
        ...

    def test_gen():
        yield

    @contextmanager
    def test_cmx():
        yield

    async def test_async():
        ...

    async def test_agen():
        yield

    @asynccontextmanager
    async def test_acm():
        yield

    class T:
        def hello(self):
            ...

    assert is_function(test_plain)
    assert is_function(test_gen)
    assert is_function(test_cmx)
    assert is_function(test_async)
    assert is_function(test_agen)
    assert is_function(test_acm)
    assert is_function(T.hello)

    assert not is_function(1)
