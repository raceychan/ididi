from contextlib import AsyncExitStack, ExitStack
from contextvars import ContextVar, Token
from functools import lru_cache, partial, wraps
from inspect import isawaitable, iscoroutinefunction
from types import MappingProxyType, TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Awaitable,
    Callable,
    ContextManager,
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
from ._itypes import (
    INSPECT_EMPTY,
    GraphConfig,
    GraphIgnoreConfig,
    IAnyFactory,
    IAsyncFactory,
    IEmptyFactory,
    IFactory,
    INode,
    INodeConfig,
    INodeFactory,
    NodeConfig,
    TDecor,
    TEntryDecor,
)
from ._type_resolve import (
    get_typed_signature,
    is_async_context_manager,
    is_class,
    is_context_manager,
    is_function,
    is_unsolvable_type,
    resolve_annotation,
    resolve_factory,
)
from .errors import (
    AsyncResourceInSyncError,
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
from .node import (
    DependentNode,
    LazyDependent,
    resolve_annotated,
    resolve_inject,
    should_override,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

Stack = TypeVar("Stack", ExitStack, AsyncExitStack)


class AbstractScope(Generic[Stack]):
    _stack: Stack
    _graph: "DependencyGraph"
    _name: Hashable
    _pre: Maybe[Union["SyncScope", "AsyncScope"]]
    _resolution_registry: ResolutionRegistry
    _registered_singleton: set[type]

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._name})"

    def cache_result(self, dependent: IEmptyFactory[T], result: T) -> None:
        self._resolution_registry.register(dependent, result)

    def get_cached(self, dependent: IEmptyFactory[T]) -> Maybe[T]:
        return self._resolution_registry.get(dependent)

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

    def register_singleton(
        self, dependent: T, dependent_type: Union[type[T], None] = None
    ) -> None:
        """
        Register a dependent to be injected
        This is particular useful for top level dependents
        """

        if dependent_type is None:
            dependent_type = type(dependent)
        self._resolution_registry.register(dependent_type, dependent)
        self._registered_singleton.add(dependent_type)


class SyncScope(AbstractScope[ExitStack]):
    __slots__ = (
        "_graph",
        "_stack",
        "_name",
        "_pre",
        "_resolution_registry",
        "_registered_singleton",
    )

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
        self._resolution_registry = ResolutionRegistry()
        self._registered_singleton = set()

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
    __slots__ = (
        "_graph",
        "_stack",
        "_name",
        "_pre",
        "_resolution_registry",
        "_registered_singleton",
    )

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
        self._resolution_registry = ResolutionRegistry()
        self._registered_singleton = set()

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

    __slots__ = ("_graph", "_scope", "_token", "_name")

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
    A Directed Acyclic Graph where each dependent is a node.

    Example
    ---
    ```python
    from typing import AsyncGenerator
    from ididi import DependencyGraph, use, entry, AsyncResource

    async def conn_factory(engine: AsyncEngine) -> AsyncGenerator[Repository, None]:
        async with engine.begin() as conn:
            yield conn

    class UnitOfWork:
        def __init__(self, conn: AsyncConnection=use(conn_factory)):
            self._conn = conn

    @entry
    async def main(command: CreateUser, uow: UnitOfWork):
        await uow.execute(build_query(command))
    ```
    """

    _nodes: GraphNodes[Any]
    "Map a type to a dependent node"
    _resolved_nodes: GraphNodes[Any]
    "Nodes that have been recursively resolved and is validated to be resolvable."
    _type_registry: TypeRegistry
    "Map a type to its implementations"

    __slots__ = (
        "_config",
        "_nodes",
        "_resolved_nodes",
        "_resolution_registry",
        "_type_registry",
        "_registered_singleton",
        "_scope_context",
    )

    def __init__(
        self,
        *,
        self_inject: bool = True,
        ignore: Maybe[GraphIgnoreConfig] = MISSING,
        partial_resolve: bool = False,
    ):
        """
        - self_inject:

          - If True, the current instance of the `DependencyGraph` will be injected into any dependency that requires it.

        - ignore:

          - A configuration that defines dependencies to be ignored during resolution.
          - This can be a tuple of `(dependency_name, dependency_type)`.
          - For example, `ignore=('name', str)` will ignore dependencies named `'name'` or those of type `str`.

        - partial_resolve:

          - If set to `True`, suppresses the `UnsolvableDependencyError` during static resolution of dependencies.
          - This assumes that unsolvable dependencies will be provided through an override later.

        ### Example:
        - Creating a DependencyGraph with self-injection enabled, ignoring 'str' type dependencies, and allowing partial resolution.

            ```python
            graph = DependencyGraph(self_inject=False, ignore=('name', str), partial_resolve=True)
            ```

        ### Notes:
        - `self_inject` is useful when you have a dependency that requires current instance of `DependencyGraph` as its dependency.
        - `ignore` is a flexible configuration to skip over specific dependencies during resolution.
        - `partial_resolve` helps to avoid errors when certain dependencies are meant to be provided with overrides.
        """
        self._config = GraphConfig(
            self_inject=self_inject, ignore=ignore, partial_resolve=partial_resolve
        )
        self._nodes = dict()
        self._resolved_nodes: GraphNodes[Any] = dict()
        self._type_registry = TypeRegistry()
        self._resolution_registry = ResolutionRegistry()
        self._registered_singleton: set[type] = set()
        self._scope_context = ContextVar[Union[SyncScope, AsyncScope]](
            "connection_context"
        )

        if self._config.self_inject:
            self.register_singleton(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    def __contains__(self, item: type) -> bool:
        return item in self._nodes

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
        return Visitor(self._nodes)

    def _merge_nodes(self, other: "DependencyGraph"):
        for dep_type, other_node in other.nodes.items():
            current_node = self._nodes.get(dep_type)
            current_resolved = self._resolved_nodes.get(dep_type)
            other_resolved = other.resolved_nodes.get(dep_type)

            if not current_node or (should_override(other_node, current_node)):
                self._nodes[dep_type] = other_node
                if other_resolved:
                    self._resolved_nodes[dep_type] = other_resolved
                continue

            # the conflict case, we only update if current node is not resolved
            if not current_resolved and other_resolved:
                self._resolved_nodes[dep_type] = other_resolved

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
            self._register_node(node)
        else:
            node = self._nodes[concrete_type]

        return node

    def _create_lazy_dependent(self, node: DependentNode[T]) -> LazyDependent[T]:
        """Create a lazy version of a node with appropriate resolver."""

        def resolver(dep_type: type[T]) -> T:
            self.static_resolve(dep_type)
            return self.resolve(dep_type)

        return LazyDependent[T](
            dependent_type=node.dependent_type,
            factory=node.factory,
            signature=node.signature,
            resolver=resolver,
        )

    def _remove_node(self, node: DependentNode[Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.pop(dependent_type, None)

    def _register_node(self, node: DependentNode[Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent_type

        dependent_type: type = get_origin(dep_type) or dep_type

        # Skip if registered
        if dependent_type in self._nodes:
            return

        # Register main type
        self._nodes[dependent_type] = node

        # Register type mappings
        self._type_registry.register(dependent_type)

    # ==========================  Public ==========================

    def merge(self, other: Union["DependencyGraph", Sequence["DependencyGraph"]]):
        """
        Merge the other graphs into this graph, update the current graph.

        ## Logic

        1. If a node not in current graph, it will be added,
        2. otherwise, compare the `resolve order` of these two nodes, keep the one with higher order.

        ### resolve order

        1. node with resource management factory > node with a plain factory.
        2. node with a plain factory > node with default constructor.
        """

        if isinstance(other, DependencyGraph):
            others = (other,)
        else:
            others = other

        for other in others:
            try:
                other._scope_context.get()
            except LookupError:
                pass
            else:
                raise MergeWithScopeStartedError()

            self._merge_nodes(other)

            self._type_registry.update(other._type_registry)
            self._resolution_registry.update(other._resolution_registry)

    def set_context_scope(
        self, scope: Union[SyncScope, AsyncScope]
    ) -> Token[Union[SyncScope, AsyncScope]]:
        return self._scope_context.set(scope)

    def reset_context_scope(self, token: Token[Union[SyncScope, AsyncScope]]):
        self._scope_context.reset(token)

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
        self._registered_singleton.clear()

        if clear_nodes:
            self._type_registry.clear()
            self._resolved_nodes.clear()
            self._nodes.clear()

    def is_registered_singleton(self, dependent_type: type) -> bool:
        return dependent_type in self._registered_singleton

    def register_singleton(
        self, dependent: T, dependent_type: Union[type[T], None] = None
    ) -> None:
        """
        Register a dependent instance to be injected, provide a dependent type if it's not the same as the instance type.
        """
        if dependent_type is None:
            dependent_type = type(dependent)
        self._resolution_registry.register(dependent_type, dependent)
        self._registered_singleton.add(dependent_type)

    def static_resolve(
        self,
        dependent: INode[P, T],
        resolve_node_config: Maybe[NodeConfig] = MISSING,
    ) -> DependentNode[T]:
        """
        Recursively analyze the type information of a dependency.
        """
        if self._config.partial_resolve is False and is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        current_path: list[type] = []

        def dfs(dependent_factory: INode[P, T]) -> DependentNode[T]:
            if is_class(dependent_factory):
                dependent_type = cast(type, dependent_factory)
            else:
                dependent_type = resolve_factory(dependent_factory)
                # statically resolve a unnoded factory function
                if (dependent_type) not in self.nodes:
                    factory = cast(INodeFactory[P, T], dependent_factory)
                    node = DependentNode[T].from_factory(
                        factory=factory, config=NodeConfig()
                    )
                    self._register_node(node)

            current_path.append(dependent_type)

            if dependent_type in self._resolved_nodes:
                return self._resolved_nodes[dependent_type]

            node = self._resolve_concrete_node(dependent_type, resolve_node_config)
            if self.is_registered_singleton(dependent_type):
                return node

            for param in node.actualized_params():
                param_type = param.param_type
                if param_type in current_path:
                    i = current_path.index(param_type)
                    cycle = current_path[i:] + [param_type]
                    raise CircularDependencyDetectedError(cycle)
                if sub_node := self._nodes.get(param_type):
                    if sub_node.config.reuse:
                        continue
                    for parent in current_path:
                        parent_config = self._nodes[parent].config
                        if parent_config.reuse:
                            raise ReusabilityConflictError(current_path, param_type)
                    continue
                if inject_node := (
                    resolve_inject(param_type) or resolve_inject(param.default)
                ):
                    self._register_node(inject_node)
                    self._resolved_nodes[param_type] = inject_node
                    continue
                if is_provided(param.default):
                    continue
                if param.unresolvable:
                    if self._config.partial_resolve:
                        continue
                    raise UnsolvableDependencyError(
                        dep_name=param.name,
                        dependent_type=dependent_type,
                        dependency_type=param.param_type,
                        factory=node.factory,
                    )

                if param_tuple := resolve_annotated(param.name, param_type):
                    _, param_type = param_tuple

                try:
                    dep_node = dfs(param_type)
                except UnsolvableNodeError as une:
                    une.add_context(
                        node.dependent_type, param.name, param.param_annotation
                    )
                    raise

                self._register_node(dep_node)
                self._resolved_nodes[param_type] = dep_node

            node.check_for_resolvability()
            self._resolved_nodes[dependent_type] = node
            current_path.pop()
            return node

        return dfs(dependent)

    def static_resolve_all(self) -> None:
        unsolved_nodes: set[type] = self._nodes.keys() - self._resolved_nodes.keys()
        for node_type in unsolved_nodes:
            self.static_resolve(node_type)

    def scope(self, name: Maybe[Hashable] = MISSING) -> ScopeProxy:
        """
        create a scope for resource,
        resources resolved wihtin the scope would be

        scope_name: give a iditifier to the scope so that you can get it with
        dg.get_scope(scope_name)
        """
        return ScopeProxy(self, name=name)

    @lru_cache(None)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        "Recursively check if a dependent type contains any resource dependency"

        if not (resolved_node := self._resolved_nodes.get(dep_type)):
            resolved_node = self.static_resolve(dep_type)

        if self.is_registered_singleton(resolved_node.dependent_type):
            return False

        if resolved_node.is_resource:
            return True

        contain_resource = any(
            (
                self.should_be_scoped(param_type)
                for _, param_type in resolved_node.unsolved_params(())
            )
        )
        return contain_resource

    @overload
    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: Literal[False] = False,
    ) -> SyncScope: ...

    @overload
    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: Literal[True],
    ) -> AsyncScope: ...

    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: bool = False,
    ) -> Union[SyncScope, AsyncScope]:
        """
        Get most recently created scope

        as_async: bool
        pure typing helper, ignored at runtime
        """
        try:
            scope = self._scope_context.get()
        except LookupError as le:
            if create_on_miss:
                return cast(Union[SyncScope, AsyncScope], self.scope(name))
            raise OutOfScopeError() from le

        if is_provided(name):
            return scope.get_scope(name)
        return scope

    def get_resolve_cache(
        self, dependent: IFactory[P, T], scope: Maybe[Any]
    ) -> Maybe[T]:
        if scope:
            if not isinstance(scope, AbstractScope):
                raise PositionalOverrideError(scope)

            if is_provided(solution := scope.get_cached(dependent)):
                return solution

            if self.should_be_scoped(dependent):
                return MISSING

        return self._resolution_registry.get(dependent)

    @overload
    def resolve(
        self,
        dependent: IAnyFactory[T],
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
        Resolves a dependent, builds all its dependencies, injects them into its factory, and returns the instance.

        ### Resolve Priority:
        1. Override: Any dependencies provided via keyword arguments will take priority.
        2. Factory: If no override is provided, the factory's resolution will be used.
        3. Default Value: If neither an override nor a factory is available, the default value is used.
        4. __init__: build dependencies according to the `__init__` method of the dependent type

        ### Example:

        Resolve a dependent with overrides and a specific scope

        ```python
        instance = dependency_graph.resolve(my_factory, scope=my_scope, my_override=override_value)
        ```

        ### Notes:
        - Ensure that any overrides are passed as keyword arguments, not positional.
        """

        if args:
            raise PositionalOverrideError(args)

        if is_provided(resolution := self.get_resolve_cache(dependent, scope)):
            return resolution

        node: DependentNode[T] = self.static_resolve(dependent)
        node_config = node.config
        resolved_params: dict[str, Any] = overrides

        ignores = self._config.ignore + tuple(resolved_params)

        for param_name, param_type in node.unsolved_params(ignores):
            if self._config.partial_resolve and is_unsolvable_type(param_type):
                continue

            if node_config.lazy:
                dep_node = self.static_resolve(param_type, node_config)
                resolved_params[param_name] = self._create_lazy_dependent(dep_node)
            else:
                resolved_params[param_name] = self.resolve(param_type, scope)

        resolved = node.inject_params(resolved_params)

        is_reuse = node_config.reuse
        if node.factory_type == "function":
            if is_reuse:
                self._resolution_registry.register(dependent, resolved)
            return cast(T, resolved)

        if is_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if is_reuse:
                scope.cache_result(dependent, instance)
        elif is_async_context_manager(resolved):
            raise AsyncResourceInSyncError(resolved)
        else:
            instance = resolved
            if is_reuse:
                self._resolution_registry.register(dependent, instance)
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

        if is_provided(resolution := self.get_resolve_cache(dependent, scope)):
            return resolution

        node: DependentNode[T] = self.static_resolve(dependent)
        resolved_params: dict[str, Any] = overrides

        ignores = self._config.ignore + tuple(resolved_params)

        for param_name, param_type in node.unsolved_params(ignores):
            if self._config.partial_resolve and is_unsolvable_type(param_type):
                continue
            resolved_params[param_name] = await self.aresolve(param_type, scope)

        resolved = node.inject_params(resolved_params)

        is_reuse = node.config.reuse
        if node.factory_type == "function":
            instance = resolved
            if isawaitable(instance):
                instance = await instance

            if is_reuse:
                self._resolution_registry.register(dependent, instance)
            return cast(T, instance)

        if is_async_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = await scope.enter_async_context(resolved)
            if is_reuse:
                scope.cache_result(dependent, instance)
        elif is_context_manager(resolved):
            if not scope:
                raise ResourceOutsideScopeError(node.dependent_type)
            instance = scope.enter_context(resolved)
            if is_reuse:
                scope.cache_result(dependent, instance)
        else:
            instance = resolved
            if is_reuse:
                self._resolution_registry.register(dependent, instance)

        return cast(T, instance)

    @overload
    def entry(self, **iconfig: Unpack[INodeConfig]) -> TEntryDecor: ...

    @overload
    def entry(
        self, func: IFactory[P, T], **iconfig: Unpack[INodeConfig]
    ) -> Callable[..., T]: ...

    def entry(
        self,
        func: Union[IFactory[P, T], IAsyncFactory[P, T], None] = None,
        **iconfig: Unpack[INodeConfig],
    ) -> Union[Callable[..., Union[T, Awaitable[T]]], TEntryDecor]:
        """
        statically resolve dependencies of the decorated function \
            then replace it with a new function, \
            where the new func will solve dependency after being called.

        ### Dependency overrides must be passed as kwarg arguments

        it is recommended to put data first then dependencies in func signature
        such as:
        
        ```py
        @entry
        async def create_user(user_name: str, user_email: str, repo: Repository):
            await repository.add_user(user_name, user_email)

        # client code
        await create_user(user_name, usere_email)
        ```

        ### - Positional only dependency is not supported, e.g.
        
        ```py
        @entry
        async def func(email_sender: EmailSender, /):
            ...

        # this would raise an exception
        ```
        """
        # TODO:
        # 1. better typing supoprt by adding more generic vars to func
        # checkout https://github.com/dbrattli/Expression/blob/main/expression/core/pipe.py

        if not func:
            configured = cast(TEntryDecor, partial(self.entry, **iconfig))
            return configured

        config = NodeConfig(**iconfig)
        should_ignore = self._config.ignore + config.ignore
        sig = get_typed_signature(func)

        depends_on_resource: bool = False
        unresolved: list[tuple[str, type]] = []
        for name, param in sig.parameters.items():
            param_type = resolve_annotation(param.annotation)
            if name in should_ignore or param_type in should_ignore:
                continue
            if is_unsolvable_type(param_type):
                continue
            if inject_node := (
                resolve_inject(param_type) or resolve_inject(param.default)
            ):
                self._register_node(inject_node)
                self._resolved_nodes[param_type] = inject_node
                param_type = inject_node.dependent_type

            self.static_resolve(param_type, config)
            # if True, stay True
            depends_on_resource = depends_on_resource or self.should_be_scoped(
                param_type
            )
            unresolved.append((name, param_type))

        if iscoroutinefunction(func):
            async_func: Callable[..., Awaitable[T]] = cast(
                Callable[..., Awaitable[T]], func
            )

            @wraps(async_func)
            async def _async_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                context_scope = self.use_scope(create_on_miss=True, as_async=True)
                async with context_scope as scope:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        resolved = await self.aresolve(param_type, scope)
                        kwargs[param_name] = resolved

                    r = await async_func(*args, **kwargs)
                    return r

            @wraps(async_func)
            async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                for param_name, param_type in unresolved:
                    if param_name in kwargs:
                        continue
                    resolved = await self.aresolve(param_type)
                    kwargs[param_name] = resolved

                r = await async_func(*args, **kwargs)
                return r

            return _async_scoped_wrapper if depends_on_resource else _async_wrapper
        else:
            sync_func: Callable[..., T] = cast(Callable[..., T], func)

            @wraps(sync_func)
            def _sync_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                context_scope = self.use_scope(create_on_miss=True, as_async=False)
                with context_scope as scope:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        resolved = self.resolve(param_type, scope)
                        kwargs[param_name] = resolved

                    r = sync_func(*args, **kwargs)
                    return r

            @wraps(sync_func)
            def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                for param_name, param_type in unresolved:
                    if param_name in kwargs:
                        continue
                    resolved = self.resolve(param_type)
                    kwargs[param_name] = resolved

                r = sync_func(*args, **kwargs)
                return r

            return _sync_scoped_wrapper if depends_on_resource else _sync_wrapper

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
            configured = cast(TDecor, partial(self.node, **config))
            return configured

        if not self._config.partial_resolve and is_unsolvable_type(factory_or_class):
            raise TopLevelBulitinTypeError(factory_or_class)

        factory_return = get_typed_signature(factory_or_class).return_annotation
        if factory_return is not INSPECT_EMPTY:
            if old_node := self._nodes.get(resolve_annotation(factory_return)):
                self._remove_node(old_node)

        node = DependentNode[T].from_node(factory_or_class, config=NodeConfig(**config))

        self._register_node(node)
        return factory_or_class
