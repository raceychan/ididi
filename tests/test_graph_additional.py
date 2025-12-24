from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from types import MethodType, SimpleNamespace
from typing import Annotated, Any, Protocol

import pytest

from ididi import Graph, Ignore, use
from ididi._node import Dependency
from ididi.config import EmptyIgnore
from ididi.errors import (
    ConfigConflictError,
    DeprecatedError,
    ParamReusabilityConflictError,
    UnsolvableNodeError,
)
from ididi.utils.param_utils import MISSING


class EntryService:
    ...


class ProtocolDependency(Protocol):
    def ping(self) -> None:
        ...


def test_resolver_name_property():
    graph = Graph(name="main")

    assert graph.name == "main"


def test_should_be_scoped_annotated_ignore_returns_false():
    graph = Graph()
    ignored_type = Ignore[int]

    assert graph.should_be_scoped(ignored_type) is False


def test_should_be_scoped_respects_node_ignore():
    graph = Graph()

    class Service:
        ...

    node = SimpleNamespace(
        dependent=Service,
        is_resource=False,
        ignore=("ignored",),
        dependencies=(Dependency(name="ignored", param_type=int, annotation=int, default=MISSING), ),
        reuse=False,
    )
    graph._analyzed_nodes[Service] = node  # type: ignore[attr-defined]

    assert graph.should_be_scoped(Service) is False





def test_analyze_params_respects_ignore(monkeypatch: pytest.MonkeyPatch):
    graph = Graph()

    def factory() -> str:
        return "value"

    annotated = Annotated[str, use(factory, reuse=False)]
    dependency = Dependency(name="dep", param_type=annotated, annotation=annotated, default=MISSING)

    def fake_build_dependencies(function, signature):
        return [dependency]

    monkeypatch.setattr("ididi._node.build_dependencies", fake_build_dependencies)

    called = {"include": 0}

    def fake_include(self, *args, **kwargs):
        called["include"] += 1
        return SimpleNamespace(dependent="ignored_type")

    monkeypatch.setattr(graph, "include_node", MethodType(fake_include, graph))
    monkeypatch.setattr(
        graph,
        "analyze",
        MethodType(
            lambda self, dependent, reuse=False, ignore=(): pytest.fail(
                "analyze should not be called"
            ),
            graph,
        ),
    )


    requires_scope, unresolved = graph.analyze_params(
        lambda dep: None, reuse=False, ignore=("ignored_type",)
    )

    assert requires_scope is False
    assert unresolved == []
    assert called["include"] == 1


def test_should_be_scoped_builtin_type_raises_unsolvable():
    graph = Graph()
    with pytest.raises(UnsolvableNodeError):
        graph.should_be_scoped(int)


def test_should_be_scoped_adds_context_on_failure():
    graph = Graph()

    class NeedsProtocol:
        def __init__(self, dependency: ProtocolDependency):
            self.dependency = dependency

    with pytest.raises(UnsolvableNodeError) as excinfo:
        graph.should_be_scoped(NeedsProtocol)

    notes = "".join(getattr(excinfo.value, "__notes__", ()))
    assert "NeedsProtocol" in notes and "dependency" in notes


def test_should_be_scoped_adds_context_when_dependency_skipped_in_analyze():
    graph = Graph()

    class Service:
        def __init__(self, dependency: ProtocolDependency = None):
            self.dependency = dependency

    with pytest.raises(UnsolvableNodeError) as excinfo:
        graph.should_be_scoped(Service)

    notes = "".join(getattr(excinfo.value, "__notes__", ()))
    assert "Service" in notes and "dependency" in notes and "ProtocolDependency" in notes


def test_analyze_params_rejects_use_defaults():
    graph = Graph()

    def entry(
        service: Annotated[EntryService, use(EntryService)] = use(EntryService),
    ) -> EntryService:
        return service

    with pytest.raises(DeprecatedError):
        graph.analyze_params(entry, reuse=MISSING, ignore=EmptyIgnore)


def test_analyze_params_reports_reuse_conflicts():
    graph = Graph()

    graph.node(EntryService, reuse=False)

    def entry(service: Annotated[EntryService, use(EntryService, reuse=True)]) -> EntryService:
        return service

    with pytest.raises(ParamReusabilityConflictError):
        graph.analyze_params(entry, reuse=MISSING, ignore=EmptyIgnore)


@pytest.mark.asyncio
async def test_entry_async_scoped_wrapper_preserves_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = Graph()
    dependency_type = object()

    def analyze_stub(self, func, reuse, ignore):
        return True, [("dep", dependency_type)]

    monkeypatch.setattr(graph, "analyze_params", MethodType(analyze_stub, graph))

    class DummyScope:
        def __init__(self):
            self.called = False

        async def resolve(self, dep_type):
            self.called = True
            return "resolved"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    dummy_scope = DummyScope()

    @asynccontextmanager
    async def fake_ascope(self, name=""):
        yield dummy_scope

    monkeypatch.setattr(graph, "ascope", MethodType(fake_ascope, graph))

    @graph.entry
    async def func(dep: str) -> str:
        return dep

    result = await func(dep="provided")

    assert result == "provided"
    assert dummy_scope.called is False


@pytest.mark.asyncio
async def test_entry_async_wrapper_preserves_kwargs(monkeypatch: pytest.MonkeyPatch):
    graph = Graph()
    dependency_type = object()

    def analyze_stub(self, func, reuse, ignore):
        return False, [("dep", dependency_type)]

    monkeypatch.setattr(graph, "analyze_params", MethodType(analyze_stub, graph))

    async def fail_aresolve(self, dep_type):
        raise AssertionError("aresolve should not be called")

    monkeypatch.setattr(graph, "aresolve", MethodType(fail_aresolve, graph))

    @graph.entry
    async def func(dep: str) -> str:
        return dep

    result = await func(dep="provided")

    assert result == "provided"


def test_entry_sync_scoped_wrapper_preserves_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    graph = Graph()
    dependency_type = object()

    def analyze_stub(self, func, reuse, ignore):
        return True, [("dep", dependency_type)]

    monkeypatch.setattr(graph, "analyze_params", MethodType(analyze_stub, graph))

    class DummyScope:
        def __init__(self):
            self.called = False

        def resolve(self, dep_type):
            self.called = True
            return "resolved"

    dummy_scope = DummyScope()

    @contextmanager
    def fake_scope(self, name=""):
        yield dummy_scope

    monkeypatch.setattr(graph, "scope", MethodType(fake_scope, graph))

    @graph.entry
    def func(dep: str) -> str:
        return dep

    result = func(dep="provided")

    assert result == "provided"
    assert dummy_scope.called is False


def test_entry_sync_wrapper_preserves_kwargs(monkeypatch: pytest.MonkeyPatch):
    graph = Graph()
    dependency_type = object()

    def analyze_stub(self, func, reuse, ignore):
        return False, [("dep", dependency_type)]

    monkeypatch.setattr(graph, "analyze_params", MethodType(analyze_stub, graph))

    def fail_resolve(self, dep_type):
        raise AssertionError("resolve should not be called")

    monkeypatch.setattr(graph, "resolve", MethodType(fail_resolve, graph))

    @graph.entry
    def func(dep: str) -> str:
        return dep

    result = func(dep="provided")

    assert result == "provided"


def test_resolve_scope_register_exit_callback_not_implemented():
    graph = Graph()
    scope = graph._create_scope("base")

    with pytest.raises(NotImplementedError):
        # call ResolveScope implementation explicitly
        super(type(scope), scope).register_exit_callback(lambda: None)


@pytest.mark.asyncio
async def test_async_scope_enter_context_handles_sync_context():
    graph = Graph()
    scope = graph._create_ascope("async")
    await scope.__aenter__()
    state: dict[str, Any] = {"entered": False, "exited": False}

    @contextmanager
    def cm():
        state["entered"] = True
        try:
            yield "value"
        finally:
            state["exited"] = True

    result = scope.enter_context(cm())
    assert result == "value"
    await scope.__aexit__(None, None, None)
    assert state["entered"] is True
    assert state["exited"] is True


@pytest.mark.asyncio
async def test_async_scope_registers_reuse_singleton():
    graph = Graph()
    scope = graph._create_ascope("async")
    await scope.__aenter__()

    class Service:
        ...

    instance = Service()
    result = await scope.aresolve_callback(
        instance, Service, "function", True
    )

    assert result is instance
    assert Service in scope._resolved_singletons  # type: ignore[attr-defined]
    assert scope._resolved_singletons[Service] is instance  # type: ignore[attr-defined]
    await scope.__aexit__(None, None, None)


def test_merge_nodes_reuse_conflict():
    graph_a = Graph()
    graph_b = Graph()

    class Service:
        ...

    graph_a.include_node(Service, reuse=False)
    graph_b.include_node(Service, reuse=True)

    with pytest.raises(ConfigConflictError):
        graph_a._merge_nodes(graph_b)
