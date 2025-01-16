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
    ClassVar,
    ContextManager,
    Generator,
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypeVar,
    Union,
    cast,
    final,
    get_origin,
    overload,
)

from typing_extensions import Self, Unpack

from ._ds import GraphNodes, GraphNodesView, ResolvedInstances, TypeRegistry, Visitor
from ._node import DependentNode, NodeConfig, resolve_use, should_override
from ._type_resolve import (
    get_bases,
    get_typed_signature,
    is_class,
    is_unsolvable_type,
    resolve_annotation,
    resolve_factory,
)
from .errors import (
    AsyncResourceInSyncError,
    CircularDependencyDetectedError,
    MergeWithScopeStartedError,
    OutOfScopeError,
    PositionalOverrideError,
    ResourceOutsideScopeError,
    ReusabilityConflictError,
    TopLevelBulitinTypeError,
    UnsolvableDependencyError,
    UnsolvableNodeError,
)
from .interfaces import (
    EntryFunc,
    GraphIgnore,
    GraphIgnoreConfig,
    IAnyFactory,
    IAsyncFactory,
    IEmptyFactory,
    IFactory,
    INode,
    INodeConfig,
    TDecor,
    TEntryDecor,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

Stack = TypeVar("Stack", ExitStack, AsyncExitStack)


def register_dependent(
    mapping: dict[Union[IEmptyFactory[T], type[T]], T],
    dependent_type: IEmptyFactory[T],
    instance: T,
):
    if isinstance(dependent_type, type):
        instance_type: type[T] = type(instance)
        for base in get_bases(instance_type):
            if base in mapping:
                continue
            mapping[base] = instance

    mapping[dependent_type] = instance


class AbstractScope(Generic[Stack]):
    _stack: Stack
    _graph: "Graph"
    _name: Hashable
    _pre: Maybe[Union["SyncScope", "AsyncScope"]]
    _resolution_registry: ResolvedInstances[Any]
    _registered_singleton: set[type]

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._name})"

    def cache_result(self, dependent: IEmptyFactory[T], result: T) -> None:
        register_dependent(self._resolution_registry, dependent, result)

    def get_cached(self, dependent: IEmptyFactory[T]) -> Maybe[T]:
        return self._resolution_registry.get(dependent, MISSING)

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
        register_dependent(self._resolution_registry, dependent_type, dependent)
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
        graph: "Graph",
        *,
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
    ):
        self._graph = graph
        self._name = name
        self._pre = pre
        self._stack = ExitStack()
        self._resolution_registry = {}
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
        graph: "Graph",
        *,
        name: Maybe[Hashable] = MISSING,
        pre: Maybe["SyncScope"] = MISSING,
    ):
        self._graph = graph
        self._name = name
        self._pre = pre
        self._stack = AsyncExitStack()
        self._resolution_registry = {}
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

    def __init__(self, graph: "Graph", name: Maybe[Hashable] = MISSING):
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


class GraphConfig:
    __slots__ = ("self_inject", "ignore")

    def __init__(self, *, self_inject: bool, ignore: Maybe[GraphIgnoreConfig]):
        self.self_inject = self_inject
        if not is_provided(ignore):
            ignore = tuple()
        elif not isinstance(ignore, tuple):
            ignore = (ignore,)

        self.ignore = ignore


@final
class Graph:
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

    _scope_context: ClassVar[ContextVar[Union[SyncScope, AsyncScope]]] = ContextVar(
        "idid_scope_ctx"
    )

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
    )

    def __init__(
        self, *, self_inject: bool = True, ignore: Maybe[GraphIgnoreConfig] = MISSING
    ):
        """
        - self_inject:

          - If True, the current instance of the `DependencyGraph` will be injected into any dependency that requires it.

        - ignore:

          - A configuration that defines dependencies to be ignored during resolution.
          - This can be a tuple of `(dependency_name, dependency_type)`.
          - For example, `ignore=('name', str)` will ignore dependencies named `'name'` or those of type `str`.



        ### Example:
        - Creating a DependencyGraph with self-injection enabled, ignoring 'str' type dependencies, and allowing partial resolution.

            ```python
            graph = DependencyGraph(self_inject=False, ignore=('name', str))
            ```

        ### Notes:
        - `self_inject` is useful when you have a dependency that requires current instance of `DependencyGraph` as its dependency.
        - `ignore` is a flexible configuration to skip over specific dependencies during resolution.
        """
        self._config = GraphConfig(self_inject=self_inject, ignore=ignore)
        self._nodes = dict()
        self._resolved_nodes: GraphNodes[Any] = dict()
        self._type_registry = TypeRegistry()
        self._resolution_registry: ResolvedInstances[Any] = {}
        self._registered_singleton: set[type] = set()

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
    def resolution_registry(self) -> ResolvedInstances[Any]:
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

    def _merge_nodes(self, other: "Graph"):
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

    def _resolve_concrete_node(self, dependent: type[T]) -> DependentNode[Any]:
        # user assign impl via factory
        if (node := self._nodes.get(dependent)) and node.factory_type != "default":
            return node

        concrete_type = self._type_registry[dependent][-1]
        concrete_node = self._nodes[concrete_type]
        concrete_node.check_for_implementations()
        return concrete_node

    def _remove_node(self, node: DependentNode[Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.pop(dependent_type, None)
        self._resolved_nodes.pop(dependent_type, None)

    def _register_node(self, node: DependentNode[Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent_type
        dependent_type: type = get_origin(dep_type) or dep_type
        if dependent_type in self._nodes:
            return
        self._nodes[dependent_type] = node
        self._type_registry.register(dependent_type)

    def _node(
        self, dependent: INode[P, T], config: Maybe[NodeConfig] = MISSING
    ) -> DependentNode[T]:
        node = DependentNode[T].from_node(dependent, config=config)

        if ori_node := self._nodes.get(node.dependent_type):
            if should_override(node, ori_node):
                self._remove_node(ori_node)

        self._register_node(node)
        return node

    def _manage_resolved(
        self,
        resolved: T,
        dependent_type: type[T],
        factory_type: str,
        is_reuse: bool,
        scope: Maybe[SyncScope],
    ):
        if factory_type in ("default", "function"):
            instance = resolved
            if is_reuse:
                register_dependent(self._resolution_registry, dependent_type, resolved)
            return instance

        if not scope:
            raise ResourceOutsideScopeError(dependent_type)

        instance = scope.enter_context(cast(ContextManager[T], resolved))
        if is_reuse:
            scope.cache_result(dependent_type, instance)
        return instance

    async def _manage_aresolved(
        self,
        resolved: Union[T, Awaitable[T]],
        dependent_type: type[T],
        factory_type: str,
        is_reuse: bool,
        scope: Maybe[AsyncScope],
    ):
        if factory_type in ("default", "function"):
            instance = await resolved if isawaitable(resolved) else resolved
            if is_reuse:
                register_dependent(self._resolution_registry, dependent_type, instance)
        else:
            if not scope:
                raise ResourceOutsideScopeError(dependent_type)

            if factory_type == "resource":
                instance = scope.enter_context(cast(ContextManager[T], resolved))
            else:
                instance = await scope.enter_async_context(
                    cast(AsyncContextManager[T], resolved)
                )
            if is_reuse:
                scope.cache_result(dependent_type, instance)

        return cast(T, instance)

    # ==========================  Public ==========================

    def merge(self, other: Union["Graph", Sequence["Graph"]]):
        """
        Merge the other graphs into this graph, update the current graph.

        ## Logic

        1. If a node not in current graph, it will be added,
        2. otherwise, compare the `resolve order` of these two nodes, keep the one with higher order.

        ### resolve order

        1. node with resource management factory > node with a plain factory.
        2. node with a plain factory > node with default constructor.
        """

        if isinstance(other, Graph):
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
        register_dependent(self._resolution_registry, dependent_type, dependent)
        self._registered_singleton.add(dependent_type)

    def remove_singleton(self, dependent_type: type) -> None:
        "Remove the registered singleton from current graph, return if not found"
        if dependent_type in self._registered_singleton:
            self._registered_singleton.remove(dependent_type)

    def remove_dependent(self, dependent_type: type) -> None:
        "Remove the dependent from current graph, return if not found"
        if not (node := self._nodes.get(dependent_type)):
            return
        self._remove_node(node)

    def check_param_conflict(self, param_type: type, current_path: list[type]):
        if param_type in current_path:
            i = current_path.index(param_type)
            cycle = current_path[i:] + [param_type]
            raise CircularDependencyDetectedError(cycle)

        if (subnode := self._nodes.get(param_type)) and not subnode.config.reuse:
            if any(self._nodes[p].config.reuse for p in current_path):
                raise ReusabilityConflictError(current_path, param_type)

    def analyze(
        self,
        dependent: INode[P, T],
        *,
        config: Maybe[NodeConfig] = MISSING,
        ignore: Maybe[GraphIgnore] = MISSING,
    ) -> DependentNode[T]:
        """
        Recursively analyze the type information of a dependency.
        """
        if node := self._resolved_nodes.get(dependent):
            return node

        if is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        current_path: list[type] = []
        ignore = (ignore + self._config.ignore) if ignore else self._config.ignore

        def dfs(
            dependent_factory: INode[P, T], dignore: GraphIgnore
        ) -> DependentNode[T]:
            if is_class(dependent_factory):
                dependent_type = cast(type, dependent_factory)
                # when we register a concrete node we also register its bases to type_registry
                if dependent_type not in self._type_registry:
                    self._node(dependent_type, config)
            else:
                dependent_type = resolve_factory(dependent_factory)
                if dependent_type not in self._nodes:
                    self._node(dependent, config)

            if dependent_type in self._resolved_nodes:
                return self._resolved_nodes[dependent_type]

            node = self._resolve_concrete_node(dependent_type)

            if self.is_registered_singleton(dependent_type):
                return node

            current_path.append(dependent_type)
            for param in node.analyze_unsolved_params(dignore):
                if (param_type := param.param_type) in self._resolved_nodes:
                    continue

                self.check_param_conflict(param_type, current_path)

                if inject_node := resolve_use(param_type):
                    self._node(inject_node.factory)
                    continue

                if is_provided(param.default):
                    continue

                if param.unresolvable:
                    raise UnsolvableDependencyError(
                        dep_name=param.name,
                        dependent_type=dependent_type,
                        dependency_type=param.param_type,
                        factory=node.factory,
                    )

                try:
                    pnode = dfs(param_type, ())
                except UnsolvableNodeError as une:
                    une.add_context(node.dependent_type, param.name, param.param_type)
                    raise

                self._register_node(pnode)
                self._resolved_nodes[param_type] = pnode

            current_path.pop()
            self._resolved_nodes[dependent_type] = node
            return node

        return dfs(dependent, ignore)

    def analyze_nodes(self) -> None:
        unsolved_nodes: set[type] = self._nodes.keys() - self._resolved_nodes.keys()
        for node_type in unsolved_nodes:
            self.analyze(node_type)

    static_resolve = analyze
    static_resolve_all = analyze_nodes

    def scope(self, name: Maybe[Hashable] = MISSING) -> ScopeProxy:
        """
        create a scope for resource,
        resources resolved wihtin the scope would be

        scope_name: give a iditifier to the scope so that you can get it with
        dg.get_scope(scope_name)
        """
        return ScopeProxy(self, name=name)

    @lru_cache(1024)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        "Recursively check if a dependent type contains any resource dependency"

        if not (resolved_node := self._resolved_nodes.get(dep_type)):
            resolved_node = self.analyze(dep_type)

        if self.is_registered_singleton(resolved_node.dependent_type):
            return False

        if resolved_node.is_resource:
            return True

        unsolved_params = resolved_node.unsolved_params(ignore=self._config.ignore)
        contain_resource = any(
            self.should_be_scoped(param_type) for _, param_type in unsolved_params
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

        return self._resolution_registry.get(dependent, MISSING)

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

        provided_params = tuple(overrides)
        node: DependentNode[T] = self.analyze(dependent, ignore=provided_params)

        if node.factory_type == "aresource":
            raise AsyncResourceInSyncError(node.dependent_type)

        unsolved_params = node.unsolved_params(self._config.ignore + provided_params)

        params = overrides

        for param_name, param_type in unsolved_params:
            params[param_name] = self.resolve(param_type, scope)

        resolved = cast(T, node.factory(**params))

        return self._manage_resolved(
            resolved=resolved,
            dependent_type=node.dependent_type,
            factory_type=node.factory_type,
            is_reuse=node.config.reuse,
            scope=scope,
        )

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

        provided_params = tuple(overrides)
        node: DependentNode[T] = self.analyze(dependent, ignore=provided_params)

        unsolved_params = node.unsolved_params(self._config.ignore + provided_params)

        for param_name, param_type in unsolved_params:
            overrides[param_name] = await self.aresolve(param_type, scope)

        resolved = cast(Union[T, Awaitable[T]], node.inject_params(overrides))
        return await self._manage_aresolved(
            resolved=resolved,
            dependent_type=node.dependent_type,
            factory_type=node.factory_type,
            is_reuse=node.config.reuse,
            scope=scope,
        )

    def _analyze_entry(self, config: NodeConfig, func: IFactory[P, T]):
        ignores = self._config.ignore + config.ignore
        func_params = get_typed_signature(func).parameters.items()
        depends_on_resource: bool = False
        unresolved: list[tuple[str, type]] = []

        for i, (name, param) in enumerate(func_params):
            param_type = resolve_annotation(param.annotation)
            if i in ignores or name in ignores or param_type in ignores:
                continue
            if is_unsolvable_type(param_type):
                continue
            if inject_node := (resolve_use(param_type) or resolve_use(param.default)):
                self._register_node(inject_node)
                self._resolved_nodes[param_type] = inject_node
                param_type = inject_node.dependent_type

            self.analyze(param_type, config=config)
            depends_on_resource = depends_on_resource or self.should_be_scoped(
                param_type
            )
            unresolved.append((name, param_type))

        return depends_on_resource, unresolved

    @overload
    def entry(self, **iconfig: Unpack[INodeConfig]) -> TEntryDecor: ...

    @overload
    def entry(
        self, func: IFactory[P, T], **iconfig: Unpack[INodeConfig]
    ) -> EntryFunc[P, T]: ...

    def entry(
        self,
        func: Union[IFactory[P, T], IAsyncFactory[P, T], None] = None,
        **iconfig: Unpack[INodeConfig],
    ) -> Union[EntryFunc[P, T], TEntryDecor]:
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
        require_scope, unresolved = self._analyze_entry(config, func)

        def replace(
            before: Maybe[type[T]] = MISSING,
            after: Maybe[type[T]] = MISSING,
            **kwargs: type[Any],
        ):
            nonlocal unresolved

            for i, (pname, ptype) in enumerate(unresolved):
                if ptype is before:
                    after = cast(type[T], after)
                    unresolved[i] = (pname, after)

                if pname in kwargs:
                    unresolved[i] = (pname, kwargs[pname])

        if iscoroutinefunction(func):
            if require_scope:

                @wraps(func)
                async def _async_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    context_scope = self.use_scope(create_on_miss=True, as_async=True)
                    async with context_scope as scope:
                        for param_name, param_type in unresolved:
                            if param_name in kwargs:
                                continue
                            resolved = await self.aresolve(param_type, scope)
                            kwargs[param_name] = resolved

                        r = await func(*args, **kwargs)
                        return r

                f = _async_scoped_wrapper
            else:

                @wraps(func)
                async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        resolved = await self.aresolve(param_type)
                        kwargs[param_name] = resolved

                    r = await func(*args, **kwargs)
                    return r

                f = _async_wrapper
        else:
            sync_func = cast(Callable[..., T], func)
            if require_scope:

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

                f = _sync_scoped_wrapper
            else:

                @wraps(sync_func)
                def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        resolved = self.resolve(param_type)
                        kwargs[param_name] = resolved

                    r = sync_func(*args, **kwargs)
                    return r

                f = _sync_wrapper

        setattr(f, replace.__name__, replace)
        return cast(EntryFunc[P, T], f)

    @overload
    def node(self, dependent: type[T], **config: Unpack[INodeConfig]) -> type[T]: ...

    @overload
    def node(
        self, dependent: IFactory[P, T], **config: Unpack[INodeConfig]
    ) -> IFactory[P, T]: ...

    @overload
    def node(self, **config: Unpack[INodeConfig]) -> TDecor: ...

    def node(
        self,
        dependent: Union[INode[P, T], None] = None,
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

        if not dependent:
            configured = cast(TDecor, partial(self.node, **config))
            return configured

        if is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        self._node(dependent, config=NodeConfig(**config))
        return dependent


DependencyGraph = Graph
