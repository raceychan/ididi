import inspect
import typing as ty
from collections import defaultdict
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from functools import lru_cache, partial
from types import MappingProxyType, TracebackType

from .errors import (
    AsyncResourceInSyncError,
    MissingImplementationError,
    NodeCreationErrorChain,
    PositionalOverrideError,
    TopLevelBulitinTypeError,
    UnsolvableNode,
)
from .node import AbstractDependent, DependentNode
from .registry import GraphNodes, GraphNodesView, ResolutionRegistry, TypeRegistry
from .type_resolve import (
    get_typed_signature,
    is_async_context_manager,
    is_context_manager,
    is_function,
    is_unresolved_type,
)
from .types import INSPECT_EMPTY, GraphConfig, IFactory, INodeConfig, NodeConfig, TDecor
from .utils.param_utils import MISSING, Maybe, is_provided

# TODO: separate scope and async scope for better typing support
"""
with dg.scope() as scope: # return sync scope
    resource = scope.resolve(Resource)
    assert resource.is_opened

async with dg.scope() as scope: # return async scope
    resource = await scope.resolve(AsyncResource)
    assert resource.is_opened
"""


class SyncScope:
    _scope: ExitStack

    __slots__ = ("_graph", "_scope")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._scope = ExitStack()

    def __enter__(self) -> "SyncScope":
        return self

    def __exit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        self._scope.__exit__(exc_type, exc_value, traceback)
        self._graph.remove_scope(self._scope)

    def resolve[T](self, dependent: type[T], /, **overrides: ty.Any) -> T:
        return self._graph.resolve(dependent, self._scope, **overrides)


class AsyncScope:
    _scope: AsyncExitStack

    __slots__ = ("_graph", "_scope")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._scope = AsyncExitStack()

    async def __aenter__(self) -> "AsyncScope":
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        await self._scope.__aexit__(exc_type, exc_value, traceback)
        self._graph.remove_scope(self._scope)

    async def resolve[T](self, dependent: type[T], /, **overrides: ty.Any) -> T:
        return await self._graph.aresolve(dependent, self._scope, **overrides)


class ScopeProxy:
    _scope: Maybe[SyncScope | AsyncScope]

    __slots__ = ("_graph", "_scope")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._scope = MISSING

    def __enter__(self) -> SyncScope:
        self._scope = SyncScope(self._graph).__enter__()
        return self._scope

    async def __aenter__(self) -> AsyncScope:
        self._scope = await AsyncScope(self._graph).__aenter__()
        return self._scope

    def __exit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        ty.cast(SyncScope, self._scope).__exit__(exc_type, exc_value, traceback)

    async def __aexit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        await ty.cast(AsyncScope, self._scope).__aexit__(exc_type, exc_value, traceback)


type ScopedResolution = defaultdict[
    ExitStack | AsyncExitStack, dict[type | ty.Callable[..., ty.Any], ty.Any]
]


class DependencyGraph:
    """
    ### Description:
    A dependency DAG (Directed Acyclic Graph) that manages dependency nodes and their relationships.

    [config]
    static_resolve: bool
    ---
    whether to statically resolve all nodes when entering the graph.
    """

    def __init__(self, static_resolve: bool = True):
        self._nodes: GraphNodes[ty.Any] = {}
        self._resolved_nodes: GraphNodes[ty.Any] = {}
        self._resolution_registry = ResolutionRegistry()
        self._scoped_resolution: ScopedResolution = defaultdict(dict)
        self._type_registry = TypeRegistry()
        self._config = GraphConfig(static_resolve=static_resolve)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    def __stats__(self) -> str:
        """
        Returns a concise statistics of the graph showing:
        - Total number of nodes
        - Number of resolved instances
        - Root nodes (nodes with no dependents)
        - Leaf nodes (nodes with no dependencies)
        """
        roots = {t for t in self._nodes if not self.get_dependent_types(t)}
        leaves = {t for t in self._nodes if not self.get_dependency_types(t)}

        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)}, "
            f"roots=[{', '.join(t.__name__ for t in sorted(roots, key=lambda x: x.__name__))}], "
            f"leaves=[{', '.join(t.__name__ for t in sorted(leaves, key=lambda x: x.__name__))}])"
        )

    async def __aenter__(self) -> "DependencyGraph":
        """
        should we reopen closed resources when entering a graph?
        """
        if not self._config.static_resolve:
            return self

        for node_type in self._nodes.copy():
            if node_type in self._resolved_nodes:
                continue
            self.static_resolve(node_type)
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: ty.Any,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def nodes(self) -> GraphNodesView[ty.Any]:
        return MappingProxyType(self._nodes)

    @property
    def resolution_registry(self) -> ResolutionRegistry:
        """
        A mapping of dependent types to their resolved instances.
        """
        return self._resolution_registry

    @property
    def type_registry(self) -> TypeRegistry:
        """
        A mapping of abstract types to their implementations.
        """
        return self._type_registry

    @property
    def resolved_nodes(self) -> GraphNodesView[ty.Any]:
        """
        A node is considered resolved if all its dependencies have been resolved.
        Which means all its forward dependencies have been resolved.
        This would also include builtin types.
        """
        return MappingProxyType(self._resolved_nodes)

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.pop(dependent_type, None)

    def get_dependent_types[T](self, dependent: type[T]) -> list[type[T]]:
        """
        Get all types that depend on the given type.
        """
        dependents: list[type[T]] = []
        for node_type, node in self._nodes.items():
            for dep_param in node.signature:
                sub_dependent: AbstractDependent[ty.Any] = dep_param.node.dependent
                if sub_dependent.dependent_type == dependent:
                    dependents.append(node_type)
        return dependents

    def get_dependency_types[T](self, dependent: type[T]) -> list[type[T]]:
        """
        Get all types that the given type depends on.
        """
        node = self._nodes[dependent]
        deps: list[type[T]] = []
        for param in node.signature:
            dependent = param.dependent_type
            deps.append(dependent)
        return deps

    def top_sorted_dependencies(self) -> list[type]:
        """
        Return types in dependency order (topological sort).
        """
        order: list[type] = []
        visited = set[type]()

        def visit(node_type: type):
            if node_type in visited:
                return

            visited.add(node_type)
            node = self._nodes[node_type]
            for dep_param in node.signature:
                dependent_type = dep_param.dependent_type
                if dependent_type not in self._nodes:
                    continue
                visit(dependent_type)

            order.append(node_type)

        for node_type in self._nodes:
            visit(node_type)

        return order

    def reset(self, clear_resolved: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_resolved: bool
        ---
        whether to clear:
        - the node registry,
        - the type registry,
        """
        self._resolution_registry.clear()
        if clear_resolved:
            self._type_registry.clear()
            self._resolved_nodes.clear()
            self._nodes.clear()

    def remove_scope(self, scope: ExitStack | AsyncExitStack) -> None:
        self._scoped_resolution.pop(scope, None)

    async def aclose(self) -> None:
        """
        Close any resources held by resolved instances.
        Closes in reverse initialization order.
        """
        close_order = reversed(self.top_sorted_dependencies())
        await self._resolution_registry.close_all(close_order)
        self.reset(clear_resolved=True)

    def _resolve_concrete_type[T](self, abstract_type: type[T]) -> type[T]:
        """
        Resolve abstract type to concrete implementation.
        """
        implementations: list[type] | None = self._type_registry.get(abstract_type)

        if not implementations:
            raise MissingImplementationError(abstract_type)

        last_implementations = implementations[-1]
        return last_implementations

    def _resolve_concrete_node[T](self, dependent: type[T]) -> DependentNode[T]:
        if registered_node := self._nodes.get(dependent):
            if is_function(registered_node.factory):
                return registered_node

        try:
            concrete_type = self._resolve_concrete_type(dependent)
        except MissingImplementationError:
            node = DependentNode.from_node(dependent, config=NodeConfig())
            self.register_node(node)
        else:
            node = self._nodes[concrete_type]
        return node

    def replace_node(
        self, old_node: DependentNode[ty.Any], new_node: DependentNode[ty.Any]
    ) -> None:
        """
        Replace an existing node with a new node.
        """
        self.remove_node(old_node)
        self.register_node(new_node)

    def __static_resolve[
        T, **P
    ](self, dependent_factory: type[T] | ty.Callable[P, T]) -> DependentNode[T]:
        """
        Resolve a dependency without building its instance.
        """
        if is_function(dependent_factory):
            sig = get_typed_signature(dependent_factory)
            dependent: type[T] = sig.return_annotation
        else:
            dependent = ty.cast(type[T], dependent_factory)

        if dependent in self._resolved_nodes:
            return self._resolved_nodes[dependent]

        if is_unresolved_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        # we need to check if the node is already registered and its factory.
        node = self._resolve_concrete_node(dependent)

        for subnode in node.iter_dependencies():
            resolved_node = self.__static_resolve(subnode.dependent_type)
            self.register_node(resolved_node)
            self._resolved_nodes[subnode.dependent_type] = resolved_node

        node.check_for_resolvability()
        self._resolved_nodes[dependent] = node
        return node

    def static_resolve[
        T, **P
    ](self, dependent: type[T] | ty.Callable[P, T]) -> DependentNode[T]:
        try:
            return self.__static_resolve(dependent)
        except UnsolvableNode as de:
            raise de
        except Exception as e:
            raise NodeCreationErrorChain(dependent, error=e, form_message=True)

    @ty.overload
    def resolve[
        **P, T
    ](
        self, dependent: ty.Callable[P, T], scope: Maybe[ExitStack] = MISSING, /
    ) -> T: ...

    @ty.overload
    def resolve[
        T, **P
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[ExitStack] = MISSING,
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    def resolve[
        **P, T
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[ExitStack] = MISSING,
        /,
        *args: ty.Any,
        **overrides: ty.Any,
    ) -> T:
        """
        Resolve a dependency and build its instance.
        NOTE: overrides will only be applied to the requested type, not recursive
        """

        if args:
            raise PositionalOverrideError(args)

        node_dep_type = dependent

        if node_dep_type in self._resolution_registry:
            return self._resolution_registry[node_dep_type]

        if is_provided(scope) and (
            solution := self._scoped_resolution[scope].get(node_dep_type)
        ):
            return solution

        node: DependentNode[T] = self.static_resolve(node_dep_type)

        resolved_deps: dict[str, ty.Any] = overrides.copy()

        for dep_param in node.signature:
            dep_name = dep_param.name
            if dep_param.should_not_resolve:
                continue

            if dep_name in overrides:
                continue

            sub_node_dep_type = dep_param.node.dependent_type

            resolved_dep = self.resolve(sub_node_dep_type, scope)
            resolved_deps[dep_name] = resolved_dep

        # TODO: make NodeResolveErrorChain
        resolved = node.build(**resolved_deps)

        if is_context_manager(resolved):
            assert is_provided(scope)
            instance = scope.enter_context(resolved)
            self._scoped_resolution[scope][node_dep_type] = instance
            return ty.cast(T, instance)
        elif is_async_context_manager(resolved):
            # since we support ACM class.
            if is_function(node.factory):
                raise AsyncResourceInSyncError(node.factory)

        if node.config.reuse:
            self._resolution_registry.register(node_dep_type, resolved)
        return ty.cast(T, resolved)

    async def aresolve[
        **P, T
    ](
        self,
        dependent: ty.Callable[P, T],
        /,
        scope: Maybe[AsyncExitStack] = MISSING,
        **overrides: ty.Any,
    ) -> T:
        """
        Async version of resolve that handles async context managers and coroutines.
        """
        node_dep_type = dependent

        if node_dep_type in self._resolution_registry:
            return self._resolution_registry[node_dep_type]

        if is_provided(scope) and (
            solution := self._scoped_resolution[scope].get(node_dep_type)
        ):
            return solution

        node: DependentNode[T] = self.static_resolve(node_dep_type)
        resolved_deps: dict[str, ty.Any] = overrides.copy()

        for dep_param in node.signature:
            dep_name = dep_param.name
            if dep_param.should_not_resolve:
                continue

            if dep_name in overrides:
                continue

            sub_node_dep_type = dep_param.node.dependent_type
            resolved_dep = await self.aresolve(sub_node_dep_type, scope)
            resolved_deps[dep_name] = resolved_dep

        resolved = node.build(**resolved_deps)

        if is_async_context_manager(resolved):
            assert is_provided(scope)
            instance = await scope.enter_async_context(resolved)
            self._scoped_resolution[scope][node_dep_type] = instance
            return ty.cast(T, instance)
        elif is_context_manager(resolved):
            assert is_provided(scope)
            instance = scope.enter_context(resolved)
            self._scoped_resolution[scope][node_dep_type] = instance
            return ty.cast(T, instance)
        elif inspect.iscoroutine(resolved):
            instance = await resolved
        else:
            instance = resolved

        if node.config.reuse:
            self._resolution_registry.register(node_dep_type, instance)
        return ty.cast(T, instance)

    @lru_cache
    def factory[**P, R](self, dependent: type[R]) -> IFactory[..., R]:
        """
        A helper function that creates a resolver(a factory) for a given type.

        Example:
        ```python
        dg = DependencyGraph()
        resolver = dg.factory(AuthService)
        ```
        """
        if dependent not in self._resolved_nodes:
            self.static_resolve(dependent)

        def resolver() -> R:
            return self.resolve(dependent)

        return resolver

    def entry[
        **P, R
    ](self, func: IFactory[P, R], /, **kwargs: ty.Unpack[INodeConfig]) -> IFactory[
        [], R
    ]:
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            scope = ExitStack()

            try:
                r = ty.cast(R, self.resolve(dep, scope, *args, **kwargs))
                return r
            finally:
                scope.__exit__(None, None, None)

        async def async_func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            scope = AsyncExitStack()
            await scope.__aenter__()

            try:
                r: R = await self.aresolve(dep, scope, *args, **kwargs)
                return r
            finally:
                await scope.__aexit__(None, None, None)

        entry_type = type(f"entry_node_{func.__name__}", (object,), {})
        dep = ty.cast(type[R], entry_type)
        sig = get_typed_signature(func)

        try:
            node = DependentNode.create(
                dependent=dep, factory=func, signature=sig, config=NodeConfig(**kwargs)
            )
        except Exception as e:
            raise NodeCreationErrorChain(func, error=e, form_message=True) from e

        self.register_node(node)

        if inspect.iscoroutinefunction(func):
            f = async_func_wrapper
        else:
            f = func_wrapper

        return ty.cast(IFactory[[], R], f)

    def scope(self) -> ScopeProxy:
        return ScopeProxy(self)

    def register_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent.dependent_type
        dependent_type: type = ty.get_origin(dep_type) or dep_type

        # Skip if registered
        if dependent_type in self._nodes:
            return

        # Register main type
        self._nodes[dependent_type] = node

        # Register type mappings
        self._type_registry.register(dependent_type)

    @ty.overload
    def node[I](self, factory_or_class: type[I]) -> type[I]: ...

    @ty.overload
    def node[**P, R](self, factory_or_class: IFactory[P, R]) -> IFactory[P, R]: ...

    @ty.overload
    def node(self, **config: ty.Unpack[INodeConfig]) -> TDecor: ...

    def node[
        **P, R
    ](
        self,
        factory_or_class: IFactory[P, R] | type[R] | None = None,
        **config: ty.Unpack[INodeConfig],
    ) -> (IFactory[P, R] | type[R] | TDecor):
        """
        ### Decorator to register a node in the dependency graph.

        This does NOT statically resolve the node

        - Can be used with both factory functions and classes.
        - If decorating a factory function that returns an existing type,
        it will override the factory for that type.
        ----
        Examples:

        #### Register a class:

        ```python
        dg = DependencyGraph()

        @dg.node
        class AuthService: ...
        ```

        #### Register a factory function:

        ```python
        @dg.node
        def auth_service_factory() -> AuthService: ...
        ```
        """

        # TODO: might make this non-recursive
        if not factory_or_class:
            configed = ty.cast(TDecor, partial(self.node, **config))
            return configed

        node_config = NodeConfig(**config)

        sig = get_typed_signature(factory_or_class)
        return_type: type[R] = sig.return_annotation

        if return_type is not INSPECT_EMPTY and return_type in self._nodes:
            old_node = self._nodes[return_type]
            self.remove_node(old_node)

        if inspect.isasyncgenfunction(factory_or_class):
            factory_or_class = asynccontextmanager(factory_or_class)
        elif inspect.isgeneratorfunction(factory_or_class):
            factory_or_class = contextmanager(factory_or_class)

        try:
            node = DependentNode.from_node(factory_or_class, config=node_config)
        except Exception as e:
            raise NodeCreationErrorChain(factory_or_class, error=e, form_message=True)

        self.register_node(node)
        return factory_or_class
