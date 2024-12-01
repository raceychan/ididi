import inspect
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from functools import partial
from types import MappingProxyType, TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    ContextManager,
    ForwardRef,
    Generator,
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
    Union,
    cast,
    get_origin,
    overload,
)

from typing_extensions import Self, Unpack

from ._ds import GraphNodes, GraphNodesView, ResolutionRegistry, TypeRegistry, Visitor
from ._itypes import (  # IAsyncResourceFactory,; IResourceFactory,
    INSPECT_EMPTY,
    GraphConfig,
    IAnyFactory,
    IAsyncFactory,
    IEmptyAsyncFactory,
    IEmptyFactory,
    IFactory,
    INode,
    INodeConfig,
    NodeConfig,
    TDecor,
)
from ._type_resolve import (
    get_typed_signature,
    is_async_context_manager,
    is_class,
    is_context_manager,
    is_function,
    is_unresolved_type,
    resolve_annotation,
    resolve_factory_return,
    resolve_forwardref,
)
from .errors import (
    AsyncResourceInSyncError,
    BuiltinTypeFactoryError,
    CircularDependencyDetectedError,
    MergeWithScopeStartedError,
    MissingImplementationError,
    OutOfScopeError,
    PositionalOverrideError,
    ResourceOutsideScopeError,
    ReusabilityConflictError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
    UnsolvableNodeError,
)
from .node import DependentNode, LazyDependent
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

Stack = TypeVar("Stack", ExitStack, AsyncExitStack)


class AbstractScope(Generic[Stack]):
    _stack: Stack
    _graph: "DependencyGraph"
    _name: Hashable
    _pre: Maybe[Union["SyncScope", "AsyncScope"]]
    resolutions: ResolutionRegistry

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._name})"

    def cache_result(self, dependent: IEmptyFactory[T], result: T) -> None:
        self.resolutions.register(dependent, result)

    def enter_context(self, context: ContextManager[T]) -> T:
        return self._stack.enter_context(context)

    def get_scope(self, name: Hashable) -> Self:
        if name == self._name:
            return self

        ptr = self
        while is_provided(ptr._pre):
            if ptr._pre._name == name:
                return cast(Self, ptr._pre)
            ptr = ptr._pre
        else:
            raise OutOfScopeError(name)


class SyncScope(AbstractScope[ExitStack]):
    __slots__ = ("_graph", "_stack", "_name", "_pre", "resolutions")

    def __init__(
        self,
        graph: "DependencyGraph",
        *,
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
    ):
        self._graph = graph
        self._name = name
        self._pre = pre
        self._stack = ExitStack()
        self.resolutions = ResolutionRegistry()

    def __enter__(self) -> "SyncScope":
        return self

    def __exit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)

    @overload
    def resolve(self, dependent: IAnyFactory[T], /) -> T: ...

    @overload
    def resolve(
        self,
        dependent: IFactory[P, Generator[T, None, None]],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    @overload
    def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **overrides: P.kwargs
    ) -> T: ...

    def resolve(
        self,
        dependent: IFactory[P, Union[T, Generator[T, None, None]]],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        # TODO: fix typing of _graph.resolve, get rid of cast
        r = self._graph.resolve(dependent, self, *args, **overrides)
        return cast(T, r)


class AsyncScope(AbstractScope[AsyncExitStack]):
    __slots__ = ("_graph", "_stack", "_name", "_pre", "resolutions")

    def __init__(
        self,
        graph: "DependencyGraph",
        *,
        name: Maybe[Hashable] = MISSING,
        pre: Maybe["SyncScope"] = MISSING,
    ):
        self._graph = graph
        self._name = name
        self._pre = pre
        self._stack = AsyncExitStack()
        self.resolutions = ResolutionRegistry()

    async def __aenter__(self) -> "AsyncScope":
        return self

    async def __aexit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

    @overload
    async def resolve(self, dependent: IAnyFactory[T], /) -> T: ...

    @overload
    async def resolve(
        self,
        dependent: IFactory[P, AsyncGenerator[T, None]],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    @overload
    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **overrides: P.kwargs
    ) -> T: ...

    async def resolve(
        self,
        dependent: IFactory[P, Union[T, AsyncGenerator[T, None]]],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        r = await self._graph.aresolve(dependent, self, *args, **overrides)
        return cast(T, r)

    async def enter_async_context(self, context: AsyncContextManager[T]) -> T:
        return await self._stack.enter_async_context(context)


class ScopeProxy:
    _scope: Maybe[Union[SyncScope, AsyncScope]]

    __slots__ = ("_graph", "_scope", "_token", "_name", "_pre")

    def __init__(self, graph: "DependencyGraph", name: Maybe[Hashable] = MISSING):
        self._graph = graph
        self._scope = MISSING
        self._name = name

    def __enter__(self) -> SyncScope:
        try:
            pre = self._graph.use_scope()
        except OutOfScopeError:
            pre = MISSING
        self._scope = SyncScope(self._graph, name=self._name, pre=pre).__enter__()
        self._token = self._graph.set_context_scope(self._scope)
        return self._scope

    async def __aenter__(self) -> AsyncScope:
        try:
            pre = self._graph.use_scope()
        except OutOfScopeError:
            pre = MISSING
        self._scope = await AsyncScope(
            self._graph, name=self._name, pre=pre
        ).__aenter__()
        self._token = self._graph.set_context_scope(self._scope)
        return self._scope

    def __exit__(
        self,
        exc_type: Union[type, None],
        exc_value: Any,
        traceback: Union[TracebackType, None],
    ) -> None:
        cast(SyncScope, self._scope).__exit__(exc_type, exc_value, traceback)
        self._graph.reset_context_scope(self._token)

    async def __aexit__(
        self,
        exc_type: Union[type, None],
        exc_value: Any,
        traceback: Union[TracebackType, None],
    ) -> None:
        await cast(AsyncScope, self._scope).__aexit__(exc_type, exc_value, traceback)
        self._graph.reset_context_scope(self._token)


class DependencyGraph:
    """
    ### Description:
    A DAG (Directed Acyclic Graph) where each dependent is a node.

    [config]
    static_resolve: bool
    ---
    whether to statically resolve all nodes when entering the graph.
    """

    __slots__ = (
        "_config",
        "_nodes",
        "_resolved_nodes",
        "_resolution_registry",
        "_type_registry",
        "_scope_context",
        "_visitor",
    )

    def __init__(self, static_resolve: bool = True):
        self._config = GraphConfig(static_resolve=static_resolve)
        self._nodes: GraphNodes[Any] = {}
        self._resolved_nodes: GraphNodes[Any] = {}
        self._resolution_registry = ResolutionRegistry()
        self._type_registry = TypeRegistry()
        self._scope_context = ContextVar[Union[SyncScope, AsyncScope]](
            "connection_context"
        )
        self._visitor = Visitor(self._nodes)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    def __contains__(self, item: type) -> bool:
        return item in self._nodes

    def static_resolve_all(self) -> None:
        if not self._config.static_resolve:
            return

        nodes = self._nodes.copy()

        for node_type in nodes:
            if node_type in self._resolved_nodes:
                continue
            self.static_resolve(node_type)

    async def __aenter__(self) -> "DependencyGraph":
        self.static_resolve_all()
        return self

    async def __aexit__(
        self,
        exc_type: Union[type, None],
        exc_value: Any,
        traceback: Union[TracebackType, None],
    ) -> None:
        await self.aclose()

    @property
    def nodes(self) -> GraphNodesView[Any]:
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
    def resolved_nodes(self) -> GraphNodesView[Any]:
        """
        A node is considered resolved if all its dependencies have been resolved.
        Which means all its forward dependencies have been resolved.
        This would also include builtin types.
        """
        return MappingProxyType(self._resolved_nodes)

    @property
    def visitor(self) -> Visitor:
        return self._visitor

    def merge(self, other: Union["DependencyGraph", Sequence["DependencyGraph"]]):
        """
        Merge the other graphs into this graph, update the current graph.
        """

        if isinstance(other, DependencyGraph):
            others = (other,)
        else:
            others = other

        for other in others:
            self._nodes.update(other.nodes)
            self._resolved_nodes.update(other._resolved_nodes)
            self._resolution_registry.update(other._resolution_registry)
            self._type_registry.update(other._type_registry)
            try:
                other._scope_context.get()
            except LookupError:
                pass
            else:
                raise MergeWithScopeStartedError()

        self._visitor = Visitor(self._nodes)

    def set_context_scope(
        self, scope: Union[SyncScope, AsyncScope]
    ) -> Token[Union[SyncScope, AsyncScope]]:
        return self._scope_context.set(scope)

    def reset_context_scope(self, token: Token[Union[SyncScope, AsyncScope]]):
        self._scope_context.reset(token)

    def remove_node(self, node: DependentNode[Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.pop(dependent_type, None)

    def reset(self, clear_nodes: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_nodes: bool
        ---
        whether to clear:
        - the node registry
        - the type registry
        """
        self._resolution_registry.clear()
        if clear_nodes:
            self._type_registry.clear()
            self._resolved_nodes.clear()
            self._nodes.clear()

    async def aclose(self) -> None:
        """
        Close any resources held by resolved instances.
        Closes in reverse initialization order.
        """
        self.reset(clear_nodes=True)

    def _resolve_concrete_type(self, abstract_type: type) -> type:
        """
        Resolve abstract type to concrete implementation.
        """
        implementations: Maybe[list[type]] = self._type_registry.get(abstract_type)

        if not is_provided(implementations):
            raise MissingImplementationError(abstract_type)

        last_implementations = implementations[-1]
        return last_implementations

    def _resolve_concrete_node(
        self, dependent: type, node_config: Maybe[NodeConfig]
    ) -> DependentNode[Any]:
        if registered_node := self._nodes.get(dependent):
            if is_function(registered_node.factory):
                return registered_node

        try:
            concrete_type = self._resolve_concrete_type(dependent)
        except MissingImplementationError:
            node = DependentNode[Any].from_node(dependent, config=node_config)
            self.register_node(node)
        else:
            node = self._nodes[concrete_type]

        return node

    def replace_node(
        self, old_node: DependentNode[T], new_node: DependentNode[T]
    ) -> None:
        """
        Replace an existing node with a new node.
        """
        self.remove_node(old_node)
        self.register_node(new_node)

    def _create_lazy_dependent(self, node: DependentNode[T]) -> LazyDependent[T]:
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

    def scope(self, name: Maybe[Hashable] = MISSING) -> ScopeProxy:
        """
        create a scope for resource,
        resources resolved wihtin the scope would be

        scope_name: give a iditifier to the scope so that you can get it with
        dg.get_scope(scope_name)
        """
        return ScopeProxy(self, name=name)

    def register_node(self, node: DependentNode[Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent.dependent_type
        dependent_type: type = get_origin(dep_type) or dep_type

        # Skip if registered
        if dependent_type in self._nodes:
            return

        # Register main type
        self._nodes[dependent_type] = node

        # Register type mappings
        self._type_registry.register(dependent_type)

    def static_resolve(
        self,
        dependent: IFactory[P, T],
        resolve_node_config: Maybe[NodeConfig] = MISSING,
    ) -> DependentNode[T]:
        """
        Resolve a dependency without building its instance.
        Args:
        dependent: type | Callable[P, T]
        node_config: NodeConfig
        """
        if is_unresolved_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        current_path: list[type] = []

        def dfs(dependent_factory: Union[type, IFactory[P, T]]) -> DependentNode[T]:
            if is_class(dependent_factory):
                dependent = cast(type, dependent_factory)
            else:
                sig = get_typed_signature(dependent_factory)
                dependent = resolve_factory_return(sig.return_annotation)
                if (dependent) not in self.nodes:
                    self.node(dependent_factory)

            current_path.append(dependent)

            if dependent in self._resolved_nodes:
                return self._resolved_nodes[dependent]

            node = self._resolve_concrete_node(dependent, resolve_node_config)
            current_config = node.config

            # TODO: we might let node return what params should be resolved
            # param with default, or set to be ignored should not be checked here at all
            ignore_params = current_config.ignore
            is_partial = current_config.partial

            for param in node.signature.actualized_params():
                param_type = param.param_type
                if param.name in ignore_params or param_type in ignore_params:
                    continue
                if param_type in current_path:
                    i = current_path.index(param_type)
                    cycle = current_path[i:] + [param_type]
                    raise CircularDependencyDetectedError(cycle)
                if sub_node := self._nodes.get(param_type):
                    sub_config = sub_node.config
                    if sub_config.reuse:
                        continue
                    for n in current_path:
                        parent_config = self._nodes[n].config
                        if parent_config.reuse:
                            raise ReusabilityConflictError(current_path, param_type)
                    continue
                if is_provided(param.default):
                    continue
                if param.unresolvable:
                    if is_partial:
                        continue
                    raise UnsolvableDependencyError(
                        dep_name=param.name,
                        required_type=param.param_type,
                        factory=node.factory,
                    )

                try:
                    dep_node = dfs(param_type)
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

        return dfs(dependent)

    @overload
    def use_scope(
        self, name: Maybe[Hashable] = MISSING, *, as_async: Literal[False] = False
    ) -> SyncScope: ...

    @overload
    def use_scope(
        self, name: Maybe[Hashable] = MISSING, *, as_async: Literal[True]
    ) -> AsyncScope: ...

    def use_scope(
        self, name: Maybe[Hashable] = MISSING, *, as_async: bool = False
    ) -> Union[SyncScope, AsyncScope]:
        """
        Get most recently created scope

        as_async: bool
        pure typing helper, ignored at runtime
        """
        try:
            scope = self._scope_context.get()
        except LookupError as le:
            raise OutOfScopeError() from le

        if is_provided(name):
            return scope.get_scope(name)
        return scope

    @overload
    def resolve(
        self,
        dependent: Callable[..., T],
        scope: Maybe[SyncScope] = MISSING,
        /,
    ) -> T: ...

    @overload
    def resolve(
        self,
        dependent: IFactory[P, T],
        scope: Maybe[SyncScope] = MISSING,
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    def resolve(
        self,
        dependent: IFactory[P, T],
        scope: Maybe[SyncScope] = MISSING,
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        """
        Resolve a dependency and build its instance.
        NOTE: overrides will only be applied to the current dependent.
        """
        if args:
            raise PositionalOverrideError(args)

        if scope:
            if not isinstance(cast(Any, scope), AbstractScope):
                raise PositionalOverrideError(args)
            if is_provided(solution := scope.resolutions.get(dependent)):
                return solution

        if is_provided(solution := self._resolution_registry.get(dependent)):
            return solution

        node: DependentNode[T] = self.static_resolve(dependent)
        current_config = node.config
        ignore_params = current_config.ignore

        resolved_params: dict[str, Any] = overrides

        for param_name, dpram in node.signature:
            if param_name in resolved_params:
                continue

            param_type = dpram.param_type

            if param_name in ignore_params or param_type in ignore_params:
                continue

            if is_provided(dpram.default):
                resolved_params[param_name] = dpram.default
                continue

            if current_config.lazy:
                dep_node = self.static_resolve(param_type, current_config)
                resolved_params[param_name] = self._create_lazy_dependent(dep_node)
                continue

            resolved_params[param_name] = self.resolve(param_type, scope)

        resolved = node.crate_dependent(resolved_params)

        if is_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if current_config.reuse:
                scope.cache_result(dependent, instance)
        elif is_async_context_manager(resolved):
            raise AsyncResourceInSyncError(node.factory)
        else:
            if current_config.reuse:
                self._resolution_registry.register(dependent, resolved)
            instance = resolved
        return cast(T, instance)

    async def aresolve(
        self,
        dependent: IFactory[P, T],
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

        if scope:
            if not isinstance(cast(Any, scope), AbstractScope):
                raise PositionalOverrideError(args)
            if is_provided(solution := scope.resolutions.get(dependent)):
                return solution

        if is_provided(solution := self._resolution_registry.get(dependent)):
            return solution

        node: DependentNode[T] = self.static_resolve(dependent)
        current_config = node.config
        ignore_params = current_config.ignore

        resolved_params: dict[str, Any] = overrides

        for param_name, dpram in node.signature:
            if param_name in resolved_params:
                continue

            param_type = dpram.param_type
            if param_name in ignore_params or param_type in ignore_params:
                continue

            if is_provided(dpram.default):
                resolved_params[param_name] = dpram.default
                continue

            resolved_params[param_name] = await self.aresolve(param_type, scope)

        resolved = node.crate_dependent(resolved_params)

        if is_async_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = await scope.enter_async_context(resolved)
            if current_config.reuse:
                scope.cache_result(dependent, instance)
        elif is_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if current_config.reuse:
                scope.cache_result(dependent, instance)
        else:
            if inspect.isawaitable(resolved):
                instance = await resolved
            else:
                instance = resolved

            if current_config.reuse:
                self._resolution_registry.register(dependent, instance)
        return cast(T, instance)

    @overload
    def factory(
        self, dependent: IFactory[P, T], *, use_async: Literal[True] = True
    ) -> IEmptyAsyncFactory[T]: ...

    @overload
    def factory(
        self, dependent: IFactory[P, T], *, use_async: Literal[False] = False
    ) -> IEmptyFactory[T]: ...

    def factory(
        self,
        dependent: Union[IFactory[P, T], IAsyncFactory[P, T]],
        *,
        use_async: bool = True,
    ) -> Union[IEmptyFactory[T], IEmptyAsyncFactory[T]]:
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

        def resolver() -> T:
            with self.scope() as scope:
                r = self.resolve(cast(Callable[..., T], dependent), scope)
                return r

        async def aresolver() -> T:
            async with self.scope() as scope:
                r = await self.aresolve(cast(Callable[..., T], dependent), scope)
                return r

        return aresolver if use_async else resolver

    @overload
    def entry(self, func: IAsyncFactory[P, T]) -> IEmptyAsyncFactory[T]: ...

    @overload
    def entry(self, func: IFactory[P, T]) -> IEmptyFactory[T]: ...

    def entry(
        self, func: Union[IFactory[P, T], IAsyncFactory[P, T]]
    ) -> Union[IEmptyFactory[T], IEmptyAsyncFactory[T]]:
        """
        TODO:
        1. add more generic vars to func
        checkout https://github.com/dbrattli/Expression/blob/main/expression/core/pipe.py
        TypeVarTuple
        Unpack[TypeVarTuple]
        2. allow ignore params, to support hook param like fastapi.Query
        """

        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with self.scope() as scope:
                r = self.resolve(dep, scope, *args, **kwargs)
                return r

        async def async_func_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with self.scope() as scope:
                r = await self.aresolve(dep, scope, *args, **kwargs)
                return cast(T, r)

        entry_type = type(f"entry_node_{func.__name__}", (object,), {})
        dep = cast(type[T], entry_type)
        sig = get_typed_signature(func)

        # directly create node without DependecyGraph.node, use func as factory
        config = NodeConfig(reuse=False, lazy=False, partial=True)
        node = DependentNode[T].create(
            dependent=dep, factory=func, signature=sig, config=config
        )

        self.register_node(node)

        if inspect.iscoroutinefunction(func):
            f = async_func_wrapper
        else:
            f = func_wrapper

        return f

    @overload
    def node(self, factory_or_class: type[T]) -> type[T]: ...

    @overload
    def node(self, factory_or_class: IFactory[P, T]) -> IFactory[P, T]: ...

    @overload
    def node(self, **config: Unpack[INodeConfig]) -> TDecor: ...

    def node(
        self,
        factory_or_class: Union[INode[P, T], None] = None,
        **config: Unpack[INodeConfig],
    ) -> Union[INode[P, T], TDecor]:
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
            configed = cast(TDecor, partial(self.node, **config))
            return configed

        node_config = NodeConfig(**config)

        if is_unresolved_type(factory_or_class):
            raise TopLevelBulitinTypeError(factory_or_class)

        sig = get_typed_signature(factory_or_class)
        return_type: Union[type[T], ForwardRef] = sig.return_annotation

        if is_unresolved_type(return_type):
            raise BuiltinTypeFactoryError(factory_or_class, return_type)

        if isinstance(return_type, ForwardRef):
            return_type = resolve_forwardref(factory_or_class, return_type)

        resolved_type = resolve_annotation(return_type)

        if resolved_type in self._nodes:
            if return_type is not INSPECT_EMPTY:
                old_node = self._nodes[resolved_type]
                self.remove_node(old_node)

        if inspect.isgeneratorfunction(factory_or_class):
            func_gen = contextmanager(factory_or_class)
            node = DependentNode[T].from_node(func_gen, config=node_config)
        elif inspect.isasyncgenfunction(factory_or_class):
            func_gen = asynccontextmanager(factory_or_class)
            node = DependentNode[T].from_node(func_gen, config=node_config)
        else:
            node = DependentNode[T].from_node(factory_or_class, config=node_config)

        self.register_node(node)
        return factory_or_class
