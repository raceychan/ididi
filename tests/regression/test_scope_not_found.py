import asyncio
import threading

from ididi.graph import Graph


def test_ascope_in_new_thread_without_token_raises_lookuperror():
    graph = Graph()
    exceptions = []

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def invoke():
            try:
                async with graph.ascope():
                    pass
            except Exception as exc:
                raise 

        loop.run_until_complete(invoke())
        loop.close()

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert any(
        isinstance(exc, LookupError) and "idid_scope_ctx" in str(exc)
        for exc in exceptions
    )