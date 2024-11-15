import inspect
import typing as ty
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from functools import partial
from types import MappingProxyType, TracebackType

from ._ds import GraphNodes, GraphNodesView, ResolutionRegistry, TypeRegistry
from ._itypes import (
    INSPECT_EMPTY,
    GraphConfig,
    IAsyncFactory,
    IFactory,
    INode,
    INodeConfig,
    NodeConfig,
    TDecor,
)
from ._type_resolve import (
    get_typed_signature,
    is_async_context_manager,
    is_context_manager,
    is_function,
    is_unresolved_type,
    resolve_annotation,
    resolve_forwardref,
)
from .errors import (
    AsyncResourceInSyncError,
    CircularDependencyDetectedError,
    MissingImplementationError,
    OutOfScopeError,
    PositionalOverrideError,
    ResourceOutsideScopeError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
    UnsolvableNodeError,
)
from .node import DependentNode, LazyDependent
from .utils.param_utils import MISSING, Maybe, is_provided


class AbstractScope[Stack: ExitStack | AsyncExitStack]:
    _stack: Stack
    _graph: "DependencyGraph"
    resolutions: ResolutionRegistry

    def cache_result[T](self, dependent: ty.Callable[..., T], result: T) -> None:
        self.resolutions.register(dependent, result)

    def enter_context[T](self, context: ty.ContextManager[T]) -> T:
        return self._stack.enter_context(context)


class SyncScope(AbstractScope[ExitStack]):
    __slots__ = ("_graph", "_stack", "resolutions")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._stack = ExitStack()
        self.resolutions = ResolutionRegistry()

    def __enter__(self) -> "SyncScope":
        return self

    def __exit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)

    @ty.overload
    def resolve[R](self, dependent: ty.Callable[..., R], /) -> R: ...

    @ty.overload
    def resolve[
        **P, R
    ](
        self, dependent: ty.Callable[P, R], /, *args: P.args, **overrides: P.kwargs
    ) -> R: ...

    def resolve[
        **P, R
    ](self, dependent: ty.Callable[P, R], /, *args: P.args, **overrides: P.kwargs) -> R:
        return self._graph.resolve(dependent, self, *args, **overrides)


class AsyncScope(AbstractScope[AsyncExitStack]):
    __slots__ = ("_graph", "_stack", "resolutions")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._stack = AsyncExitStack()
        self.resolutions = ResolutionRegistry()

    async def __aenter__(self) -> "AsyncScope":
        return self

    async def __aexit__(
        self,
        exc_type: type[Exception] | None,
        exc_value: Exception | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

    @ty.overload
    async def resolve[R](self, dependent: ty.Callable[..., R], /) -> R: ...

    @ty.overload
    async def resolve[
        **P, R
    ](
        self, dependent: ty.Callable[P, R], /, *args: P.args, **overrides: P.kwargs
    ) -> R: ...

    async def resolve[
        **P, R
    ](self, dependent: ty.Callable[P, R], /, *args: P.args, **overrides: P.kwargs) -> R:
        return await self._graph.aresolve(dependent, self, *args, **overrides)

    async def enter_async_context[T](self, context: ty.AsyncContextManager[T]) -> T:
        return await self._stack.enter_async_context(context)


class ScopeProxy:
    _scope: Maybe[SyncScope | AsyncScope]

    __slots__ = ("_graph", "_scope", "_token")

    def __init__(self, graph: "DependencyGraph"):
        self._graph = graph
        self._scope = MISSING

    def __enter__(self) -> SyncScope:
        self._scope = SyncScope(self._graph).__enter__()
        self._token = self._graph.set_context_scope(self._scope)
        return self._scope

    async def __aenter__(self) -> AsyncScope:
        self._scope = await AsyncScope(self._graph).__aenter__()
        self._token = self._graph.set_context_scope(self._scope)
        return self._scope

    def __exit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        ty.cast(SyncScope, self._scope).__exit__(exc_type, exc_value, traceback)
        self._graph.reset_context_scope(self._token)

    async def __aexit__(
        self, exc_type: type | None, exc_value: ty.Any, traceback: TracebackType | None
    ) -> None:
        await ty.cast(AsyncScope, self._scope).__aexit__(exc_type, exc_value, traceback)
        self._graph.reset_context_scope(self._token)


class Visitor:
    def __init__(self, nodes: GraphNodes[ty.Any]):
        self._nodes = nodes

    def get_dependents(self, dependency: type) -> list[type]:
        """
        return dependents of a dependency, both need to be in the graph
        """
        visited: set[type] = set()
        dependents: list[type] = []

        def dfs(dep: type):
            if dep in visited:
                return
            visited.add(dep)
            node = self._nodes[dep]
            for _, dep_param in node.signature:
                param_type = dep_param.param_type
                if param_type not in self._nodes:
                    continue
                if param_type is dependency:
                    dependents.append(dep)
                dfs(param_type)

        for dep in self._nodes:
            dfs(dep)
        return dependents

    def get_dependencies(self, dependent: type, recursive: bool = False) -> list[type]:
        """
        return dependencies of a dependent, both need to be in the graph

        recursive: bool; if true, return recursive dependencies.
        """
        if not recursive:
            return [p.param_type for _, p in self._nodes[dependent].signature]

        visited: set[type] = set()
        dependencies: list[type] = []

        def dfs(dep: type):
            if dep in visited:
                return
            visited.add(dep)
            node = self._nodes[dep]
            for _, dep_param in node.signature:
                param_type = dep_param.param_type
                dependencies.append(param_type)
                if param_type not in self._nodes:
                    continue
                dfs(param_type)

        dfs(dependent)
        return dependencies


class DependencyGraph:
    """
    ### Description:
    A DAG (Directed Acyclic Graph) where each dependent is a node.

    [config]
    static_resolve: bool
    ---
    whether to statically resolve all nodes when entering the graph.
    """

    def __init__(
        self,
        static_resolve: bool = True,
    ):
        self._config = GraphConfig(static_resolve=static_resolve)
        self._nodes: GraphNodes[ty.Any] = {}
        self._resolved_nodes: GraphNodes[ty.Any] = {}
        self._resolution_registry = ResolutionRegistry()
        self._type_registry = TypeRegistry()
        self._scope_context = ContextVar[SyncScope | AsyncScope]("connection_context")
        self._visitor = Visitor(self._nodes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    def static_resolve_all(self) -> None:
        if not self._config.static_resolve:
            return

        for node_type in self._nodes.copy():
            if node_type in self._resolved_nodes:
                continue
            self.static_resolve(node_type)

    async def __aenter__(self) -> "DependencyGraph":
        self.static_resolve_all()
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

    @property
    def visitor(self) -> Visitor:
        return self._visitor

    def set_context_scope(
        self, scope: SyncScope | AsyncScope
    ) -> Token[SyncScope | AsyncScope]:
        return self._scope_context.set(scope)

    def reset_context_scope(self, token: Token[SyncScope | AsyncScope]):
        self._scope_context.reset(token)

    @ty.overload
    def use_scope(self, as_async: ty.Literal[False] = False) -> SyncScope: ...

    @ty.overload
    def use_scope(self, as_async: ty.Literal[True]) -> AsyncScope: ...

    def use_scope(self, as_async: bool = False) -> SyncScope | AsyncScope:
        """
        Get most recently created scope

        as_async: bool
        pure typing helper, ignored at runtime
        """
        try:
            scope = self._scope_context.get()
        except LookupError as le:
            raise OutOfScopeError() from le
        return scope

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.pop(dependent_type, None)

    def top_sorted_dependencies(self) -> list[type]:
        """
        Return types in dependency order (topological sort).
        """
        # TODO: use visitor

        order: list[type] = []
        visited = set[type]()

        def dfs(node_type: type):
            if node_type in visited:
                return

            visited.add(node_type)
            node = self._nodes[node_type]
            for _, dep_param in node.signature:
                dependent_type = dep_param.param_type
                if dependent_type not in self._nodes:
                    continue
                dfs(dependent_type)

            order.append(node_type)

        for node_type in self._nodes:
            dfs(node_type)

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
        implementations: Maybe[list[type[T]]] = self._type_registry.get(abstract_type)

        if not is_provided(implementations):
            raise MissingImplementationError(abstract_type)

        last_implementations = implementations[-1]
        return last_implementations

    def _resolve_concrete_node[
        T
    ](self, dependent: type[T], node_config: Maybe[NodeConfig]) -> DependentNode[T]:
        if registered_node := self._nodes.get(dependent):
            if is_function(registered_node.factory):
                return registered_node

        try:
            concrete_type = self._resolve_concrete_type(dependent)
        except MissingImplementationError:
            node_config = node_config if is_provided(node_config) else NodeConfig()
            node = DependentNode.from_node(dependent, config=node_config)
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

    def _create_lazy_node[T](self, node: DependentNode[T]) -> LazyDependent[T]:
        """Create a lazy version of a node with appropriate resolver."""

        def resolver(dep_type: type[T]) -> T:
            self.static_resolve(dep_type)
            return self.resolve(dep_type)

        return LazyDependent(
            dependent_type=node.dependent_type,
            factory=node.factory,
            signature=node.signature,
            resolver=resolver,
        )

    def static_resolve[
        T, **P
    ](
        self,
        dependent: type[T] | ty.Callable[P, T],
        node_config: Maybe[NodeConfig] = MISSING,
    ) -> DependentNode[T]:
        """
        Resolve a dependency without building its instance.
        Args:
        dependent: type[T] | ty.Callable[P, T]
        node_config: NodeConfig
        """
        if is_unresolved_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        current_path: list[type] = []

        def dfs(
            dependent_factory: type[T] | ty.Callable[P, T],
            node_config: Maybe[NodeConfig],
        ) -> DependentNode[T]:

            if is_function(dependent_factory):
                sig = get_typed_signature(dependent_factory)
                dependent: type[T] = sig.return_annotation
                if dependent not in self.nodes:
                    # create node and register
                    self.node(dependent_factory)
            else:
                dependent = ty.cast(type[T], dependent_factory)

            if dependent in self._resolved_nodes:
                return self._resolved_nodes[dependent]

            node = self._resolve_concrete_node(dependent, node_config)

            current_path.append(dependent)

            for param in node.signature.actualized_params():
                param_type = param.param_type
                if param_type in current_path:
                    i = current_path.index(param_type)
                    cycle = current_path[i:] + [param_type]
                    raise CircularDependencyDetectedError(cycle)
                if param_type in self._nodes:  # shared dependency
                    continue
                if is_provided(param.default):
                    continue
                if param.unresolvable:
                    raise UnsolvableDependencyError(
                        dep_name=param.name,
                        required_type=param.param_type,
                        factory=node.factory,
                    )

                try:
                    dep_node = dfs(param_type, node_config)
                except UnsolvableNodeError as une:
                    une.add_context(
                        node.dependent_type, param.name, param.param_annotation
                    )
                    raise

                self.register_node(dep_node)
                self._resolved_nodes[param_type] = dep_node

            node.check_for_resolvability()
            self._resolved_nodes[dependent] = node
            current_path.pop()
            return node

        return dfs(dependent, node_config)

    @ty.overload
    def resolve[
        **P, T
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[SyncScope] = MISSING,
        /,
    ) -> T: ...

    @ty.overload
    def resolve[
        T, **P
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[SyncScope] = MISSING,
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    def resolve[
        **P, T
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[SyncScope] = MISSING,
        /,
        *args: ty.Any,
        **overrides: ty.Any,
    ) -> T:
        """
        Resolve a dependency and build its instance.
        NOTE: overrides will only be applied to the current dependent.
        """

        if args:
            raise PositionalOverrideError(args)

        if is_provided(scope):
            if not isinstance(ty.cast(ty.Any, scope), AbstractScope):
                raise PositionalOverrideError(args)
            if is_provided(solution := scope.resolutions.get(dependent)):
                return solution

        if is_provided(solution := self._resolution_registry.get(dependent)):
            return solution

        node: DependentNode[T] = self.static_resolve(dependent)

        resolved_args: dict[str, ty.Any] = {}

        for param_name, dpram in node.signature:
            param_type = dpram.param_type

            if param_name in overrides:
                resolved_args[param_name] = overrides[param_name]
                continue

            if is_provided(dpram.default):
                resolved_args[param_name] = dpram.default
                continue

            if node.config.lazy:
                dep_node = self.static_resolve(param_type, node.config)
                resolved_args[param_name] = self._create_lazy_node(dep_node)
                continue

            resolved_args[param_name] = self.resolve(param_type, scope)

        bound_args = node.signature.bind_arguments(resolved_args)
        resolved = node.factory(**bound_args.arguments)

        if is_context_manager(resolved):
            if not is_provided(scope):
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if node.config.reuse:
                scope.cache_result(dependent, instance)
            return ty.cast(T, instance)
        elif is_async_context_manager(resolved):
            if is_function(node.factory):  # ACM class
                raise AsyncResourceInSyncError(node.factory)

        if node.config.reuse:
            self._resolution_registry.register(dependent, resolved)
        return ty.cast(T, resolved)

    async def aresolve[
        **P, T
    ](
        self,
        dependent: ty.Callable[P, T],
        scope: Maybe[AsyncScope] = MISSING,
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        """
        Async version of resolve that handles async context managers and coroutines.
        """

        if args:
            raise PositionalOverrideError(args)

        if is_provided(scope):
            if not isinstance(ty.cast(ty.Any, scope), AbstractScope):
                raise PositionalOverrideError(args)
            if is_provided(solution := scope.resolutions.get(dependent)):
                return solution

        if is_provided(solution := self._resolution_registry.get(dependent)):
            return solution

        node: DependentNode[T] = self.static_resolve(dependent)

        resolved_args: dict[str, ty.Any] = {}

        for param_name, dpram in node.signature:
            param_type = dpram.param_type
            if param_name in overrides:
                resolved_args[param_name] = overrides[param_name]
                continue

            if is_provided(dpram.default):
                resolved_args[param_name] = dpram.default
                continue

            resolved_args[param_name] = await self.aresolve(param_type, scope)

        bound_args = node.signature.bind_arguments(resolved_args)
        resolved = node.factory(**bound_args.arguments)

        if is_async_context_manager(resolved):
            if not is_provided(scope):
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = await scope.enter_async_context(resolved)
            if node.config.reuse:
                scope.cache_result(dependent, instance)
            return ty.cast(T, instance)

        if is_context_manager(resolved):
            if not is_provided(scope):
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if node.config.reuse:
                scope.cache_result(dependent, instance)
            return ty.cast(T, instance)

        if inspect.isawaitable(resolved):
            instance = await resolved
        else:
            instance = resolved

        if node.config.reuse:
            self._resolution_registry.register(dependent, instance)
        return ty.cast(T, instance)

    @ty.overload
    def factory[
        R
    ](
        self, dependent: ty.Callable[..., R], *, use_async: ty.Literal[True] = True
    ) -> IAsyncFactory[..., R]: ...

    @ty.overload
    def factory[
        R
    ](
        self, dependent: ty.Callable[..., R], *, use_async: ty.Literal[False] = False
    ) -> IFactory[..., R]: ...

    def factory[
        R
    ](self, dependent: ty.Callable[..., R], *, use_async: bool = True) -> (
        IFactory[..., R] | IAsyncFactory[..., R]
    ):
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
            with self.scope() as scope:
                return self.resolve(dependent, scope)

        async def aresolver() -> R:
            async with self.scope() as scope:
                return await self.aresolve(dependent, scope)

        return aresolver if use_async else resolver

    def entry[**P, R](self, func: IFactory[P, R]) -> IFactory[[], R]:
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with self.scope() as scope:
                r = self.resolve(dep, scope, *args, **kwargs)
                return r

        async def async_func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            async with self.scope() as scope:
                r = await self.aresolve(dep, scope, *args, **kwargs)
                return r

        entry_type = type(f"entry_node_{func.__name__}", (object,), {})
        dep = ty.cast(type[R], entry_type)
        sig = get_typed_signature(func)

        # directly create node without DependecyGraph.node, use func as factory
        node = DependentNode.create(
            dependent=dep, factory=func, signature=sig, config=NodeConfig()
        )

        self.register_node(node)

        if inspect.iscoroutinefunction(func):
            f = async_func_wrapper
        else:
            f = func_wrapper

        return ty.cast(IFactory[[], R], f)

    def scope(self) -> ScopeProxy:
        """
        create a scope for resource,
        resources resolved wihtin the scope would be
        """
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
        factory_or_class: INode[P, R] | None = None,
        **config: ty.Unpack[INodeConfig],
    ) -> (INode[P, R] | TDecor):
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

        if not factory_or_class:
            configed = ty.cast(TDecor, partial(self.node, **config))
            return configed

        node_config = NodeConfig(**config)

        sig = get_typed_signature(factory_or_class)
        return_type: type[R] | ty.ForwardRef = sig.return_annotation

        if isinstance(return_type, ty.ForwardRef):
            return_type = resolve_forwardref(factory_or_class, return_type)

        resolved_type = resolve_annotation(return_type)

        if resolved_type in self._nodes:
            if return_type is not INSPECT_EMPTY:
                old_node = self._nodes[resolved_type]
                self.remove_node(old_node)

        if inspect.isgeneratorfunction(factory_or_class):
            func_gen = contextmanager(factory_or_class)
            node = DependentNode.from_node(func_gen, config=node_config)
        elif inspect.isasyncgenfunction(factory_or_class):
            func_gen = asynccontextmanager(factory_or_class)
            node = DependentNode.from_node(func_gen, config=node_config)
        else:
            node = DependentNode.from_node(factory_or_class, config=node_config)

        self.register_node(node)
        return factory_or_class
