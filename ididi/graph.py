from contextlib import AsyncExitStack, ExitStack
from contextvars import ContextVar, Token
from functools import lru_cache, partial, wraps
from inspect import isawaitable, iscoroutinefunction
from types import MappingProxyType, TracebackType
from typing import (
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    ClassVar,
    ContextManager,
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    cast,
    final,
    get_origin,
    overload,
)

from typing_extensions import Self, Unpack, override

from ._ds import GraphNodes, GraphNodesView, ResolvedInstances, TypeRegistry, Visitor
from ._node import (
    DefaultConfig,
    Dependencies,
    DependentNode,
    NodeConfig,
    resolve_use,
    should_override,
)
from ._type_resolve import (
    FactoryType,
    get_bases,
    get_typed_signature,
    is_class,
    is_new_type,
    is_unsolvable_type,
    resolve_annotation,
    resolve_factory,
    resolve_node_type,
)
from .config import CacheMax, EmptyIgnore, GraphConfig
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
    INodeFactory,
    TDecor,
    TEntryDecor,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

Stack = TypeVar("Stack", ExitStack, AsyncExitStack)
ScopeContext = ContextVar[Union["SyncScope", "AsyncScope"]]
ScopeToken = Token[Union["SyncScope", "AsyncScope"]]


def register_dependent(
    mapping: dict[Union[IEmptyFactory[T], type[T]], T],
    dependent_type: IEmptyFactory[T],
    instance: T,
) -> None:
    if isinstance(dependent_type, type):
        base_types = get_bases(type(instance))
        for base in base_types:
            if base in mapping:
                continue
            mapping[base] = instance

    mapping[dependent_type] = instance


class SharedData(TypedDict):
    "Data shared between graph and scope"

    nodes: GraphNodes
    resolved_nodes: dict[Hashable, DependentNode[Any]]
    type_registry: TypeRegistry
    ignore: GraphIgnore


SharedSlots: tuple[str, ...] = (
    "_nodes",
    "_resolved_nodes",
    "_type_registry",
    "_ignore",
    "_resolved_instances",
    "_registered_singletons",
)


class Resolver:
    __slots__ = SharedSlots

    def __init__(
        self,
        resolved_instances: ResolvedInstances[Any],
        registered_singletons: set[type],
        **args: Unpack[SharedData],
    ):
        self._nodes = args["nodes"]
        self._resolved_nodes = args["resolved_nodes"]
        self._type_registry = args["type_registry"]
        self._ignore = args["ignore"]

        self._registered_singletons = registered_singletons
        self._resolved_instances = resolved_instances

    def is_registered_singleton(self, dependent_type: type) -> bool:
        return dependent_type in self._registered_singletons

    @lru_cache(CacheMax)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        "Recursively check if a dependent type contains any resource dependency"
        if not (resolved_node := self._resolved_nodes.get(dep_type)):
            resolved_node = self.analyze(dep_type)

        if self.is_registered_singleton(resolved_node.dependent_type):
            return False

        if resolved_node.is_resource:
            return True

        unsolved_params = resolved_node.unsolved_params(ignore=self._ignore)
        contain_resource = any(
            self.should_be_scoped(param_type) for _, param_type in unsolved_params
        )
        return contain_resource

    def _remove_node(self, node: DependentNode[Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolved_instances.pop(dependent_type, None)
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

    def _resolve_concrete_node(self, dependent: type[T]) -> DependentNode[Any]:
        # when user assign impl via factory
        if (node := self._nodes.get(dependent)) and node.factory_type != "default":
            return node

        concrete_type = self._type_registry[dependent][-1]
        concrete_node = self._nodes[concrete_type]
        concrete_node.check_for_implementations()
        return concrete_node

    def get_resolve_cache(self, dependent: IFactory[P, T]) -> Maybe[T]:
        return self._resolved_instances.get(dependent, MISSING)

    def resolve_callback(
        self,
        resolved: Any,
        dependent_type: type[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T:
        if factory_type not in ("default", "function"):
            raise ResourceOutsideScopeError(dependent_type)

        if is_reuse:
            register_dependent(self._resolved_instances, dependent_type, resolved)
        return resolved

    async def aresolve_callback(
        self,
        resolved: Any,
        dependent_type: type[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ):
        if factory_type not in ("default", "function"):
            raise ResourceOutsideScopeError(dependent_type)

        instance = await resolved if isawaitable(resolved) else resolved
        if is_reuse:
            register_dependent(self._resolved_instances, dependent_type, instance)
        return cast(T, instance)

    @overload
    def resolve(
        self,
        dependent: IAnyFactory[T],
        /,
    ) -> T: ...

    @overload
    def resolve(
        self,
        dependent: IFactory[P, T],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    def resolve(
        self,
        dependent: IFactory[P, T],
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

        if is_provided(resolution := self.get_resolve_cache(dependent)):
            return resolution

        provided_params = frozenset(overrides)
        node: DependentNode[T] = self.analyze(dependent, ignore=provided_params)
        unsolved_params = node.unsolved_params(self._ignore | provided_params)

        params = overrides
        for param_name, param_type in unsolved_params:
            params[param_name] = self.resolve(param_type)

        resolved = node.factory(**params)
        return self.resolve_callback(
            resolved=resolved,
            dependent_type=node.dependent_type,
            factory_type=node.factory_type,
            is_reuse=node.config.reuse,
        )

    async def aresolve(
        self,
        dependent: IFactory[P, T],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T:
        """
        Async version of resolve that handles async context managers and coroutines.
        """
        if args:
            raise PositionalOverrideError(args)

        if is_provided(resolution := self.get_resolve_cache(dependent)):
            return resolution

        provided_params = frozenset(overrides)
        node: DependentNode[T] = self.analyze(dependent, ignore=provided_params)
        unsolved_params = node.unsolved_params(self._ignore | provided_params)

        for param_name, param_type in unsolved_params:
            overrides[param_name] = await self.aresolve(param_type)

        resolved = node.inject_params(overrides)
        return await self.aresolve_callback(
            resolved=resolved,
            dependent_type=node.dependent_type,
            factory_type=node.factory_type,
            is_reuse=node.config.reuse,
        )

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
        config: NodeConfig = DefaultConfig,
        ignore: Union[GraphIgnore, None] = None,
    ) -> DependentNode[T]:
        """
        Recursively analyze the type information of a dependency.
        """
        if node := self._resolved_nodes.get(dependent):
            return node

        if is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        def dfs(dependent_type: type[T], dignore: GraphIgnore) -> DependentNode[T]:
            # when we register a concrete node we also register its bases to type_registry
            if dependent_type in self._resolved_nodes:
                return self._resolved_nodes[dependent_type]

            if dependent_type not in self._type_registry:
                self._node(dependent_type, config)

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
                    pnode = dfs(param_type, EmptyIgnore)
                except UnsolvableNodeError as une:
                    une.add_context(node.dependent_type, param.name, param.param_type)
                    raise

                self._register_node(pnode)
                self._resolved_nodes[param_type] = pnode

            current_path.pop()
            self._resolved_nodes[dependent_type] = node
            return node

        if is_class(dependent) or is_new_type(dependent):
            dependent_type = resolve_annotation(dependent)
        else:
            dependent_type = resolve_factory(dependent)
            if dependent_type not in self._nodes:
                self._node(dependent, config=config)

        dependent_type = cast(type[T], dependent_type)
        current_path: list[type] = []

        ignore = (ignore | self._ignore) if ignore else self._ignore
        return dfs(dependent_type, ignore)

    def analyze_nodes(self) -> None:
        unsolved_nodes: set[type] = self._nodes.keys() - self._resolved_nodes.keys()
        for node_type in unsolved_nodes:
            self.analyze(node_type)

    def _node(
        self, dependent: INode[P, T], config: NodeConfig = DefaultConfig
    ) -> DependentNode[T]:

        node = DependentNode[T].from_node(dependent, config=config)

        if ori_node := self._nodes.get(node.dependent_type):
            if should_override(node, ori_node):
                self._remove_node(ori_node)

        self._register_node(node)
        return node

    @overload
    def node(self, dependent: type[T], **config: Unpack[INodeConfig]) -> type[T]: ...

    @overload
    def node(
        self, dependent: INodeFactory[P, T], **config: Unpack[INodeConfig]
    ) -> INodeFactory[P, T]: ...

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

    static_resolve = analyze
    static_resolve_all = analyze_nodes


@final
class Graph(Resolver):
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

    __slots__ = SharedSlots

    _scope_context: ClassVar[ScopeContext] = ContextVar("idid_scope_ctx")

    _nodes: GraphNodes
    "Map a type to a dependent node"
    _resolved_nodes: dict[Hashable, DependentNode[Any]]
    "Nodes that have been recursively resolved and is validated to be resolvable."
    _type_registry: TypeRegistry
    "Map a type to its implementations"
    _resolved_instances: ResolvedInstances[Any]
    "Instances of reusable types"
    _registered_singletons: set[type]
    "Types that are menually added singletons "

    def __init__(
        self, *, self_inject: bool = True, ignore: GraphIgnoreConfig = EmptyIgnore
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
        config = GraphConfig(self_inject=self_inject, ignore=ignore)

        super().__init__(
            nodes=dict(),
            resolved_nodes=dict(),
            type_registry=TypeRegistry(),
            resolved_instances=dict(),
            registered_singletons=set(),
            ignore=config.ignore,
        )

        if self_inject:
            self.register_singleton(self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_instances)})"
        )

    def __contains__(self, item: INode[P, T]) -> bool:
        return resolve_node_type(item) in self._nodes

    @property
    def nodes(self) -> GraphNodesView[Any]:
        return MappingProxyType(self._nodes)

    @property
    def resolution_registry(self) -> ResolvedInstances[Any]:
        """
        A mapping of dependent types to their resolved instances.
        """
        return self._resolved_instances

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
            self._resolved_instances.update(other._resolved_instances)

    def reset(self, clear_nodes: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_nodes: bool
        ---
        whether to clear:
        - the node registry
        - the type registry
        """
        self._resolved_instances.clear()
        self._registered_singletons.clear()

        if clear_nodes:
            self._type_registry.clear()
            self._resolved_nodes.clear()
            self._nodes.clear()

    def register_singleton(
        self, dependent: T, dependent_type: Union[type[T], None] = None
    ) -> None:
        """
        Register a dependent instance to be injected, provide a dependent type if it's not the same as the instance type.
        """
        if dependent_type is None:
            dependent_type = type(dependent)
        register_dependent(self._resolved_instances, dependent_type, dependent)
        self._registered_singletons.add(dependent_type)

    def remove_singleton(self, dependent_type: type) -> None:
        "Remove the registered singleton from current graph, return if not found"
        if dependent_type in self._registered_singletons:
            self._registered_singletons.remove(dependent_type)

    def remove_dependent(self, dependent: INode[P, T]) -> None:
        "Remove the dependent from current graph, return if not found"
        dependent_type = resolve_node_type(dependent)

        if node := self._nodes.get(dependent_type):
            self._remove_node(node)

    def override(self, old_dep: INode[P, T], new_dep: INode[P, T]) -> None:
        """
        globally override a dependency
        """

        self.remove_dependent(old_dep)
        self._node(new_dep)

    def search_node(self, dep_name: str) -> Union[DependentNode[Any], None]:
        for dep, node in self._nodes.items():
            if dep.__name__ == dep_name:
                return node

    def scope(self, name: Maybe[Hashable] = MISSING) -> "ScopeManager":
        """
        create a scope for resource,
        resources resolved wihtin the scope would be

        scope_name: give a iditifier to the scope so that you can get it with
        dg.get_scope(scope_name)
        """
        shared_data = SharedData(
            nodes=self._nodes,
            resolved_nodes=self._resolved_nodes,
            type_registry=self._type_registry,
            ignore=self._ignore,
        )

        return ScopeManager(
            name=name,
            scope_ctx=self._scope_context,
            graph_resolutions=self._resolved_instances,
            graph_singletons=self._registered_singletons,
            shared_data=shared_data,
        )

    @overload
    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: Literal[False] = False,
    ) -> "SyncScope": ...

    @overload
    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: Literal[True],
    ) -> "AsyncScope": ...

    def use_scope(
        self,
        name: Maybe[Hashable] = MISSING,
        *,
        create_on_miss: bool = False,
        as_async: bool = False,
    ) -> Union["SyncScope", "AsyncScope"]:
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

    def analyze_params(
        self, func: Callable[P, T], **iconfig: Unpack[INodeConfig]
    ) -> tuple[bool, list[tuple[str, type]]]:
        config = NodeConfig(**iconfig)
        ignores = self._ignore | config.ignore

        deps = Dependencies.from_signature(
            signature=get_typed_signature(func), factory=func
        )

        depends_on_resource: bool = False
        unresolved: list[tuple[str, type]] = []

        for name, dep in deps.filter_ignore(ignores):
            param_type = dep.param_type

            if is_unsolvable_type(param_type):
                continue

            if inject_node := (resolve_use(param_type) or resolve_use(dep.default)):
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
        self, func: Callable[P, T], **iconfig: Unpack[INodeConfig]
    ) -> EntryFunc[P, T]: ...

    def entry(
        self,
        func: Union[Callable[P, T], IAsyncFactory[P, T], None] = None,
        **iconfig: Unpack[INodeConfig],
    ) -> Union[EntryFunc[P, T], TEntryDecor]:
        """
        statically resolve dependencies of the decorated function \
            then replace it with a new function, \
            where the new func will solve dependency after being called.

        ### Dependency overrides must be passed as kwarg arguments

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

        require_scope, unresolved = self.analyze_params(func, **iconfig)

        def replace(
            before: Maybe[type[T]] = MISSING,
            after: Maybe[type[T]] = MISSING,
            **kw_overrides: type[Any],
        ):
            nonlocal unresolved

            for i, (pname, ptype) in enumerate(unresolved):
                if ptype is before and is_provided(after):
                    analyzed = self.analyze(after).dependent_type
                    unresolved[i] = (pname, analyzed)

                if pname in kw_overrides:
                    analyzed = self.analyze(kw_overrides[pname]).dependent_type
                    unresolved[i] = (pname, analyzed)

        if iscoroutinefunction(func):
            if require_scope:

                @wraps(func)
                async def _async_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    context_scope = self.use_scope(create_on_miss=True, as_async=True)
                    async with context_scope as scope:
                        for param_name, param_type in unresolved:
                            if param_name in kwargs:
                                continue
                            kwargs[param_name] = await scope.resolve(param_type)

                        r = await func(*args, **kwargs)
                        return r

                f = _async_scoped_wrapper
            else:

                @wraps(func)
                async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        kwargs[param_name] = await self.aresolve(param_type)
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
                            kwargs[param_name] = scope.resolve(param_type)

                        r = sync_func(*args, **kwargs)
                        return r

                f = _sync_scoped_wrapper
            else:

                @wraps(sync_func)
                def _sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    for param_name, param_type in unresolved:
                        if param_name in kwargs:
                            continue
                        kwargs[param_name] = self.resolve(param_type)
                    r = sync_func(*args, **kwargs)
                    return r

                f = _sync_wrapper

        setattr(f, replace.__name__, replace)
        return cast(EntryFunc[P, T], f)


ScopeSlots = ("_name", "_stack", "_pre", "_graph_resolutions", "_graph_singletons")


class ScopeBase(Generic[Stack]):
    __slots__ = ()
    _stack: Stack
    _name: Hashable
    _pre: Maybe[Union["SyncScope", "AsyncScope"]]

    _resolved_instances: ResolvedInstances[Any]
    _registered_singletons: set[type]

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self._name})"

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
        register_dependent(self._resolved_instances, dependent_type, dependent)
        self._registered_singletons.add(dependent_type)


class SyncScope(ScopeBase[ExitStack], Resolver):
    __slots__ = ScopeSlots + SharedSlots

    def __init__(
        self,
        *,
        graph_resolutions: ResolvedInstances[Any],
        graph_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):

        self._name = name
        self._pre = pre
        self._stack = ExitStack()
        self._graph_resolutions = graph_resolutions
        self._graph_singletons = graph_singletons
        super().__init__(**args, registered_singletons=set(), resolved_instances=dict())

    def __enter__(self) -> "SyncScope":
        return self

    def __exit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)

    @override
    def resolve_callback(
        self,
        resolved: T,
        dependent_type: type[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ):
        if factory_type in ("default", "function"):
            if is_reuse:
                register_dependent(self._graph_resolutions, dependent_type, resolved)
            return resolved

        if factory_type == "aresource":
            raise AsyncResourceInSyncError(dependent_type)

        instance = self.enter_context(cast(ContextManager[T], resolved))
        if is_reuse:
            register_dependent(self._resolved_instances, dependent_type, instance)
        return instance

    @override
    def get_resolve_cache(self, dependent: IFactory[P, T]) -> Maybe[T]:
        if resolution := self._resolved_instances.get(dependent, MISSING):
            return resolution

        if self.should_be_scoped(dependent):
            return MISSING

        return self._graph_resolutions.get(dependent, MISSING)


class AsyncScope(ScopeBase[AsyncExitStack], Resolver):
    __slots__ = ScopeSlots + SharedSlots

    def __init__(
        self,
        *,
        graph_resolutions: ResolvedInstances[Any],
        graph_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        self._name = name
        self._pre = pre
        self._stack = AsyncExitStack()
        self._graph_resolutions = graph_resolutions
        self._graph_singletons = graph_singletons

        super().__init__(**args, registered_singletons=set(), resolved_instances=dict())

    async def __aenter__(self) -> "AsyncScope":
        return self

    async def __aexit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

    @override
    def get_resolve_cache(self, dependent: IFactory[P, T]) -> Maybe[T]:
        if resolution := self._resolved_instances.get(dependent, MISSING):
            return resolution

        if self.should_be_scoped(dependent):
            return MISSING

        return self._graph_resolutions.get(dependent, MISSING)

    async def enter_async_context(self, context: AsyncContextManager[T]) -> T:
        return await self._stack.enter_async_context(context)

    @overload
    async def resolve(self, dependent: IAnyFactory[T], /) -> T: ...

    @overload
    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T: ...

    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return await super().aresolve(dependent, *args, **kwargs)

    @override
    async def aresolve_callback(
        self,
        resolved: Union[T, Awaitable[T]],
        dependent_type: type[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T:
        if factory_type in ("default", "function"):
            instance = await resolved if isawaitable(resolved) else resolved
            if is_reuse:
                register_dependent(self._graph_resolutions, dependent_type, instance)
            return cast(T, instance)

        if factory_type == "resource":
            instance = self.enter_context(cast(ContextManager[T], resolved))
        else:
            instance = await self.enter_async_context(
                cast(AsyncContextManager[T], resolved)
            )

        if is_reuse:
            register_dependent(self._resolved_instances, dependent_type, instance)
        return instance


class ScopeManager:
    __slots__ = (
        "graph_resolutions",
        "graph_singletons",
        "shared_data",
        "scope_ctx",
        "name",
        "_scope",
        "_token",
    )

    shared_data: SharedData
    scope_ctx: ScopeContext

    def __init__(
        self,
        shared_data: SharedData,
        scope_ctx: ScopeContext,
        graph_resolutions: ResolvedInstances[Any],
        graph_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
    ):
        self.shared_data = shared_data
        self.scope_ctx = scope_ctx
        self.graph_resolutions = graph_resolutions
        self.graph_singletons = graph_singletons

        self.name = name
        self._scope: Maybe[Union[SyncScope, AsyncScope]] = MISSING
        self._token: Maybe[ScopeToken] = MISSING

    def __enter__(self) -> SyncScope:
        try:
            pre = self.scope_ctx.get()
        except LookupError:
            pre = MISSING

        self._scope = SyncScope(
            graph_resolutions=self.graph_resolutions,
            graph_singletons=self.graph_singletons,
            name=self.name,
            pre=pre,
            **self.shared_data,
        ).__enter__()
        self._token = self.scope_ctx.set(self._scope)
        return self._scope

    async def __aenter__(self) -> AsyncScope:
        try:
            pre = cast(AsyncScope, self.scope_ctx.get())
        except LookupError:
            pre = MISSING

        self._scope = await AsyncScope(
            graph_resolutions=self.graph_resolutions,
            graph_singletons=self.graph_singletons,
            name=self.name,
            pre=pre,
            **self.shared_data,
        ).__aenter__()

        self._token = self.scope_ctx.set(self._scope)
        return self._scope

    def __exit__(
        self,
        exc_type: Union[type, None],
        exc_value: Any,
        traceback: Union[TracebackType, None],
    ) -> None:
        cast(SyncScope, self._scope).__exit__(exc_type, exc_value, traceback)
        if is_provided(self._token):
            self.scope_ctx.reset(self._token)

    async def __aexit__(
        self,
        exc_type: Union[type, None],
        exc_value: Any,
        traceback: Union[TracebackType, None],
    ) -> None:
        await cast(AsyncScope, self._scope).__aexit__(exc_type, exc_value, traceback)

        if is_provided(self._token):
            self.scope_ctx.reset(self._token)


DependencyGraph = Graph
