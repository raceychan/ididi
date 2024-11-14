import inspect
import typing as ty
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from functools import partial
from types import MappingProxyType, TracebackType

from ._ds import (
    GraphNodes,
    GraphNodesView,
    GraphVisitor,
    ResolutionRegistry,
    TypeRegistry,
)
from ._itypes import (
    INSPECT_EMPTY,
    GraphConfig,
    IAsyncFactory,
    IFactory,
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
    PositionalOverrideError,
    ResourceOutsideScopeError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
    UnsolvableNodeError,
)
from .node import DependencyParam, DependentNode, LazyDependent
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
        self._type_registry = TypeRegistry()
        self._config = GraphConfig(static_resolve=static_resolve)
        self._visitor = GraphVisitor[type]()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    async def __aenter__(self) -> "DependencyGraph":
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

    @property
    def visitor(self) -> GraphVisitor[type]:
        return self._visitor

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

        def visit(node_type: type):
            if node_type in visited:
                return

            visited.add(node_type)
            node = self._nodes[node_type]
            for dep_param in node.signature:
                dependent_type = dep_param.param_type
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
            self._visitor.clear()

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

    def _actualize_forwardrefs(self, node: DependentNode[ty.Any]):
        dependent_type = node.dependent_type
        for dpram in node.signature:
            param_type = dpram.param_type
            if isinstance(param_type, ty.ForwardRef):
                param_type = resolve_forwardref(node.factory, param_type)

                node.signature.update(
                    dpram.name,
                    DependencyParam(
                        name=dpram.name,
                        param=dpram.param,
                        param_annotation=dpram.param_annotation,
                        param_type=param_type,
                        default=dpram.default,
                    ),
                )
            self._visitor.add_edge(dependent_type, param_type)
            circular_path = self._visitor.find_circular_dependency()
            if circular_path:
                raise CircularDependencyDetectedError(circular_path)

    def __static_resolve[
        T, **P
    ](
        self,
        dependent_factory: type[T] | ty.Callable[P, T],
        node_config: Maybe[NodeConfig],
    ) -> DependentNode[T]:
        """
        Resolve a dependency without building its instance.
        """

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
        self._actualize_forwardrefs(node)

        for param in node.signature:
            dep_type = param.param_type
            if dep_type in self._nodes:
                continue

            if is_provided(param.default):
                continue

            if param.unresolvable:
                raise UnsolvableDependencyError(
                    dep_name=param.name,
                    required_type=param.param_type,
                    factory=node.factory,
                )

            # Recursively resolve and register dependency
            try:
                dep_node = self.__static_resolve(dep_type, node_config)
            except UnsolvableNodeError as une:
                une.add_context(node.dependent_type, param.name, param.param_annotation)
                raise

            self.register_node(dep_node)
            self._resolved_nodes[dep_type] = dep_node

        node.check_for_resolvability()
        self._resolved_nodes[dependent] = node
        return node

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
        return self.__static_resolve(dependent, node_config)

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

        # TODO:
        # if not is_provided(scope):
        #     self.global_scope.get()

        node_dep_type = dependent

        if is_provided(scope) and is_provided(
            solution := scope.resolutions.get(node_dep_type)
        ):
            return solution

        if is_provided(solution := self._resolution_registry.get(node_dep_type)):
            return solution

        node: DependentNode[T] = self.static_resolve(node_dep_type)

        resolved_args: dict[str, ty.Any] = {}

        for dpram in node.signature:
            param_name = dpram.name
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
                scope.resolutions.register(node_dep_type, instance)
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
        scope: Maybe[AsyncScope] = MISSING,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        """
        Async version of resolve that handles async context managers and coroutines.
        """
        if args:
            raise PositionalOverrideError(args)

        node_dep_type = dependent

        if is_provided(scope) and is_provided(
            solution := scope.resolutions.get(node_dep_type)
        ):
            return solution

        if is_provided(solution := self._resolution_registry.get(node_dep_type)):
            return solution

        node: DependentNode[T] = self.static_resolve(node_dep_type)

        resolved_args: dict[str, ty.Any] = {}

        for dpram in node.signature:
            param_name = dpram.name
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

            resolved_args[param_name] = await self.aresolve(param_type, scope)

        bound_args = node.signature.bind_arguments(resolved_args)
        resolved = node.factory(**bound_args.arguments)

        if is_async_context_manager(resolved):
            if not is_provided(scope):
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = await scope.enter_async_context(resolved)
            if node.config.reuse:
                scope.resolutions.register(node_dep_type, instance)
            return ty.cast(T, instance)

        if is_context_manager(resolved):
            if not is_provided(scope):
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if node.config.reuse:
                scope.resolutions.register(node_dep_type, instance)
            return ty.cast(T, instance)

        if inspect.isawaitable(resolved):
            instance = await resolved
        else:
            instance = resolved

        if node.config.reuse:
            self._resolution_registry.register(node_dep_type, instance)
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

    def entry[
        **P, R
    ](self, func: IFactory[P, R], /, **kwargs: ty.Unpack[INodeConfig]) -> IFactory[
        [], R
    ]:
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
            dependent=dep, factory=func, signature=sig, config=NodeConfig(**kwargs)
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
        """
        we might use contextvar to save scope
        so that user can have global scope

        e.g. fastapi
        async with dg.scope():
             yield

        # with dg.scope():
        #     dg.resolve(Resource)
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

        if not factory_or_class:
            configed = ty.cast(TDecor, partial(self.node, **config))
            return configed

        node_config = NodeConfig(**config)

        sig = get_typed_signature(factory_or_class)
        return_type: type[R] = sig.return_annotation

        """
        TODO: factory for multiple protocol
        class Closable(ty.Protocol): ...
        class Resource(ty.Protocol): ...

        @dg.node
        def resource_factory() -> Closable | Resource
        
        """

        resolved_type = resolve_annotation(return_type)

        if resolved_type in self._nodes:
            if return_type is not INSPECT_EMPTY:
                old_node = self._nodes[resolved_type]
                self.remove_node(old_node)
                self._visitor.remove_dependent(resolved_type)

        if inspect.isgeneratorfunction(factory_or_class):
            factory_or_class = contextmanager(factory_or_class)
        elif inspect.isasyncgenfunction(factory_or_class):
            factory_or_class = asynccontextmanager(factory_or_class)

        node = DependentNode.from_node(factory_or_class, config=node_config)
        self.register_node(node)
        return factory_or_class
