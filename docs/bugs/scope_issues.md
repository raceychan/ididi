# Scope ContextVar LookupError

- **Issue**: After upgrading ididi (1.8.x), requests hit `LookupError: <ContextVar name='idid_scope_ctx' ...>` inside `Graph.ascope`/`use_scope` when handling routes in new event loop tasks. Stack shows `_SCOPE_CONTEXT.get()` raising before any scope is set (see uvicorn ASGI trace from `/api/v1/token` POST).
- **Root cause identified**: In 1.4.4 refactor the graph/scope ContextVar became module-level without a default fallback; `Graph.ascope()` and `use_scope()` call `_SCOPE_CONTEXT.get()` directly, so new tasks without an inherited context crash.
- **Progress**:
  1. Reproduced by inspecting logs; error originates at `ididi/graph.py:373` (`_SCOPE_CONTEXT.get()`).
  2. Traced history: pre-1.4.4 used a class-level context var and seeded scope; 1.4.4 removed default handling.
  3. Drafted a fix: store `_default_scope` in `Graph`, add `_get_current_scope()` that seeds the ContextVar when missing, and update `scope()/ascope()/use_scope()` to use it. Added regression tests ensuring `use_scope`/`ascope` work when ContextVar is unset.
- **Status**: User paused the fix; changes exist locally in `ididi/graph.py` and `tests/test_scope.py` (tests pass). Decide whether to keep this fallback or roll back the 1.4.4 scope change.
- **Next options**: (a) Publish patch release with fallback; (b) Revert to pre-1.4.4 context handling; or (c) enforce explicit scope seeding in app startup.


# ididi scope LookupError（ContextVar `idid_scope_ctx`）

## 现象
- 线上请求 `POST /api/v1/token` 返回 500，日志报：`LookupError: <ContextVar name='idid_scope_ctx'>`。
- 调用栈：Starlette/Lihil 路由 → `Graph.ascope()` → `_SCOPE_CONTEXT.get()`，因为没有 token 直接抛错。
- 本地单进程/单 worker 环境没有复现。
- 当前依赖：ididi 1.8.2（纯 Python graph 实现）。1.7.x 在现有运行方式下未暴露此问题。

## 原因推断
- 从 ididi 1.4.4 起，`Graph.ascope()/use_scope()` 直接从 ContextVar 取当前 scope，不再隐式种默认 scope。若所在事件循环/任务没有预先 set 过 token，则第一次调用就会抛 `LookupError`。
- 本地单 loop/单 worker：启动时 set 的 token 沿用同一任务，问题被掩盖。
- 线上多 worker/多事件循环（或某些新建线程/loop 的调用路径）中，新的 loop 没有继承最初的 token，于是首次进入路由就触发异常。

## 已做处理（临时兜底）
- 在应用启动（lifespan）时显式为当前事件循环种默认 scope，避免没有 token：
  - 文件：`backend/src/ports/http/lifespan.py`
  - 位置：`await app.graph.aresolve(init_deps)` 之后
  - 代码：`_SCOPE_CONTEXT.set(app.graph.use_scope(DefaultScopeName))`
- 目的：保证每个 worker/事件循环在处理请求前，`idid_scope_ctx` 都有初始值，`Graph.ascope()` 不再因缺 token 500。

## 风险与后续建议
- 如果后续在新线程/新事件循环里直接调用 DI（未经过 lifespan 的 loop），仍可能缺 token，需要在这些入口也显式 set。
- 若线上使用多 worker 的预加载模式（例如 gunicorn preload_app），确保每个 worker 都执行 lifespan，否则仍有遗漏。
- 更稳妥的上游修复：在 ididi 内部为 `_SCOPE_CONTEXT.get()` 提供默认值，或在 `Graph.ascope()`/`use_scope()` 内自动 seed；或者暂时 pin 回 1.7.x。
- 本地 pytest 目前被 `pytest_lazyfixture` 插件报错卡住，未能通过测试复现该问题。

## 复现提示
- 线上：多 worker/多 loop 场景下，直接请求 `POST /api/v1/token` 触发 500（ContextVar 缺失）。
- 本地：单 worker 启动通常不会复现；可用 `uvicorn --workers N` 或在新事件循环里调用路由尝试复现。

## 关于新任务/线程触发问题的补充
- ContextVar 的 token 不会跨线程/新事件循环自动继承；`graph.ascope()` 进入时直接 `_SCOPE_CONTEXT.get()`，若当前上下文未 set 过 token 就抛 `LookupError`。
- 在主 loop 以外创建的新任务（`run_in_executor`、自建 loop、后台线程等）如果调用 `graph.ascope()`，必须先手动 `set` scope，或者用 `copy_context().run(...)` 复制已有 ContextVar，再调用 `ascope()`。
- 实际线上场景：多 worker/多 loop 或任务分发导致新 loop 没有 token，一旦调用 `ascope()` 就暴露该 bug。

## 本地复现步骤
1) 确认使用 ididi ≥ 1.4.4（当前 1.8.2）。
2) 用多 worker 启动：`cd backend && uvicorn src.ports.http.main:app_factory --factory --workers 2`。
3) 打一个最简单的请求（例如 `curl -X POST http://localhost:8000/api/v1/token`）。在未种 scope 的 worker 上会 500，日志出现 `LookupError: <ContextVar name='idid_scope_ctx'>`。
4) 或者在 REPL/脚本里新建一个事件循环/线程直接调用 `graph.ascope()`，不提前 set `_SCOPE_CONTEXT`，同样抛 `LookupError`。
5) 单测佐证：`backend/tests/test_ididi_scope_contextvar.py` 在新线程/新事件循环中调用 `graph.ascope()`，会捕获同样的 `LookupError`（未显式 set ContextVar）。示例代码：
   ```python
   import asyncio, threading
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
                   exceptions.append(exc)

           loop.run_until_complete(invoke())
           loop.close()

       t = threading.Thread(target=worker)
       t.start()
       t.join()

       assert any(
           isinstance(exc, LookupError) and "idid_scope_ctx" in str(exc)
           for exc in exceptions
       )
   ```
