from asyncio import AbstractEventLoop, get_running_loop
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token, copy_context
from functools import lru_cache, partial, wraps
from inspect import isawaitable, iscoroutinefunction
from types import MappingProxyType, MethodType, TracebackType
from typing import (  # final,; Generic,
    Annotated,
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Container,
    ContextManager,
    Final,
    Hashable,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Self, Unpack, override

from ._ds import GraphNodes, GraphNodesView, ResolvedSingletons, TypeRegistry, Visitor
from ._node import (
    DefaultConfig,
    Dependencies,
    DependentNode,
    NodeConfig,
    resolve_use,
    should_override,
)
from ._type_resolve import (
    get_bases,
    get_typed_signature,
    is_function,
    is_new_type,
    is_unsolvable_type,
    resolve_annotation,
    resolve_factory,
    resolve_node_type,
)
from .config import CacheMax, DefaultScopeName, EmptyIgnore, FactoryType, GraphConfig
from .errors import (
    AsyncResourceInSyncError,
    CircularDependencyDetectedError,
    MergeWithScopeStartedError,
    NotSupportedError,
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
    IAsyncFactory,
    IDependent,
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

from ._node cimport Dependency, DependentNode

SharedSlots: tuple[str, ...] = (
    "_nodes",
    "_analyzed_nodes",
    "_type_registry",
    "_ignore",
    "_resolved_singletons",
    "_registered_singletons",
    "_workers",
)
ScopeSlots = SharedSlots + ("_name", "_stack", "_pre")
AsyncScopeSlots = SharedSlots + ScopeSlots + ("_loop",)
GraphSlots = SharedSlots + ("_token",)

Stack = TypeVar("Stack", ExitStack, AsyncExitStack)
AnyScope = Union["SyncScope", "AsyncScope"]
ScopeToken = Token[AnyScope]
ScopeContext = ContextVar[AnyScope]

_SCOPE_CONTEXT: Final[ScopeContext] = ContextVar("idid_scope_ctx")


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




cdef object _resolve_dfs(
    Resolver resolver,
    dict nodes,
    dict cache,
    object ptype,
    dict overrides,
):
    cdef dict params
    cdef str name
    cdef Dependency param
    cdef DependentNode pnode
    cdef object resolution

    if resolution := cache.get(ptype):
        return resolution

    pnode = nodes.get(ptype) or resolver.analyze(ptype)
    params = overrides
    pdeps = pnode.dependencies

    for name, param in pdeps.items():
        if name in params:
            continue
        params[name] = _resolve_dfs(resolver, nodes, cache, param.param_type, {})

    instance = pnode.factory(**params)
    result = resolver.resolve_callback(
        instance,
        pnode.dependent,
        pnode.factory_type,
        pnode.config.reuse,
    )
    return result


async def _aresolve_dfs(
    graph: "Resolver",
    nodes: GraphNodes[Any],
    cache: ResolvedSingletons[Any],
    ptype: IDependent[T],
    overrides: dict[str, Any],
) -> T:
    if resolution := cache.get(ptype):
        return resolution

    pnode = nodes.get(ptype) or graph.analyze(ptype)

    params = overrides
    for name, param in pnode.dependencies.items():
        if name in params:
            continue
        params[name] = await _aresolve_dfs(graph, nodes, cache, param.param_type, {})

    instance = pnode.factory(**params)
    resolved = await graph.aresolve_callback(
        resolved=instance,
        dependent=pnode.dependent,
        factory_type=pnode.factory_type,
        is_reuse=pnode.config.reuse,
    )
    return resolved


@asynccontextmanager
async def syncscope_in_thread(
    loop: AbstractEventLoop, workers: ThreadPoolExecutor, cm: ContextManager[T]
) -> AsyncGenerator[T, None]:
    exc_type, exc, tb = None, None, None
    ctx = copy_context()
    cm_enter = partial(ctx.run, cm.__enter__)
    cm_exit = partial(ctx.run, cm.__exit__)

    try:
        yield await loop.run_in_executor(workers, cm_enter)
    except Exception as e:
        exc_type, exc, tb = type(e), e, e.__traceback__
        raise
    finally:
        await loop.run_in_executor(workers, cm_exit, exc_type, exc, tb)


class SharedData(TypedDict):
    "Data shared between graph and scope"

    nodes: GraphNodes[Any]
    analyzed_nodes: dict[IDependent[Any], DependentNode]
    type_registry: TypeRegistry
    ignore: GraphIgnore
    workers: ThreadPoolExecutor


cdef class Resolver:
    __slots__ = SharedSlots

    def __init__(
        self,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        **args: Unpack[SharedData],
    ):
        self._nodes = args["nodes"]
        self._analyzed_nodes = args["analyzed_nodes"]
        self._type_registry = args["type_registry"]
        self._ignore = args["ignore"]
        self._workers = args["workers"]

        self._registered_singletons = registered_singletons
        self._resolved_singletons = resolved_singletons

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_singletons)})"
        )

    def get(self, dep: INode[P, T]) -> Union[DependentNode, None]:
        if node := self._nodes.get(dep):
            return node
        return self._nodes.get(resolve_node_type(dep))

    @property
    def nodes(self) -> GraphNodesView[Any]:
        return MappingProxyType(self._nodes)

    @property
    def resolution_registry(self) -> ResolvedSingletons[Any]:
        """
        A mapping of dependent types to their resolved instances.
        """
        return self._resolved_singletons

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
        return MappingProxyType(self._analyzed_nodes)

    @property
    def visitor(self) -> Visitor:
        return Visitor(self._nodes)

    def __contains__(self, item: INode[P, T]) -> bool:
        return resolve_node_type(item) in self._nodes

    def _remove_node(self, node: DependentNode) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolved_singletons.pop(dependent_type, None)
        self._analyzed_nodes.pop(dependent_type, None)

    def _register_node(self, node: DependentNode) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent
        dependent: IDependent[Any] = get_origin(dep_type) or dep_type
        if dependent in self._nodes:
            return

        self._nodes[dependent] = node
        self._type_registry.register(dependent)

    def _resolve_concrete_node(self, dependent: IDependent[T]) -> DependentNode:
        # when user assign impl via factory
        if (node := self._nodes.get(dependent)) and node.factory_type != "default":
            return node

        concrete_type = self._type_registry[dependent][-1]
        concrete_node = self._nodes[concrete_type]
        concrete_node.check_for_implementations()
        return concrete_node

    def _analyze_dep(self, dependent: INode[P, T], config: NodeConfig) -> IDependent[T]:
        if isinstance(dependent, MethodType):
            dep_method = dependent
            bound_obj = dep_method.__self__
            if not isinstance(bound_obj, type):
                raise NotSupportedError("Instance method is not supported")
            self._node(dep_method)
            dependent = cast(IDependent[T], bound_obj)
            resolved_type = resolve_annotation(dependent)
        elif is_function(dependent):
            if is_new_type(dependent):
                resolved_type = resolve_annotation(dependent)
            else:
                if dependent not in self._nodes:
                    self._node(dependent, config)
                resolved_type = resolve_factory(dependent)
        else:
            resolved_type = resolve_annotation(dependent)

        annt_dep = None

        if get_origin(resolved_type) is Annotated:
            annt_dep = resolved_type

        if annt_dep:
            if use_meta := resolve_use(annt_dep):
                ufunc, uconfig = use_meta
                self._node(ufunc, uconfig)

            if is_function(dependent):
                node = DependentNode.from_node(dependent, config=config)
                self._nodes[dependent] = node
                return cast(IDependent[T], dependent)

            resolved_type, *_ = get_args(annt_dep)
        return resolved_type

    def _create_scope(
        self, name: Hashable = "", pre: Maybe[AnyScope] = MISSING
    ) -> "SyncScope":
        shared_data = SharedData(
            nodes=self._nodes,
            analyzed_nodes=self._analyzed_nodes,
            type_registry=self._type_registry,
            ignore=self._ignore,
            workers=self._workers,
        )
        scope = SyncScope(
            registered_singletons=self._registered_singletons.copy(),
            resolved_singletons=self._resolved_singletons.copy(),
            name=name,
            pre=pre,
            **shared_data,
        )
        return scope

    def _create_ascope(
        self, name: Hashable = "", pre: Maybe[AnyScope] = MISSING
    ) -> "AsyncScope":
        shared_data = SharedData(
            nodes=self._nodes,
            analyzed_nodes=self._analyzed_nodes,
            type_registry=self._type_registry,
            ignore=self._ignore,
            workers=self._workers,
        )
        scope = AsyncScope(
            registered_singletons=self._registered_singletons.copy(),
            resolved_singletons=self._resolved_singletons.copy(),
            name=name,
            pre=pre,
            **shared_data,
        )
        return scope

    @contextmanager
    def scope(self, name: Hashable = ""):
        previous_scope = _SCOPE_CONTEXT.get()
        scope = self._create_scope(name, previous_scope)
        token = _SCOPE_CONTEXT.set(scope)

        exc_type, exc, tb = None, None, None

        try:
            yield scope
        except Exception as e:
            exc_type, exc, tb = type(e), e, e.__traceback__
            raise
        finally:
            scope.__exit__(exc_type, exc, tb)
            _SCOPE_CONTEXT.reset(token)

    @asynccontextmanager
    async def ascope(self, name: Hashable = ""):
        previous_scope = _SCOPE_CONTEXT.get()
        ascope = self._create_ascope(name, previous_scope)
        token = _SCOPE_CONTEXT.set(ascope)

        exc_type, exc, tb = None, None, None

        try:
            yield ascope
        except Exception as e:
            exc_type, exc, tb = type(e), e, e.__traceback__
            raise
        finally:
            await ascope.__aexit__(exc_type, exc, tb)
            _SCOPE_CONTEXT.reset(token)

    # =================  Public =================

    def is_registered_singleton(self, dependent_type: IDependent[T]) -> bool:
        return dependent_type in self._registered_singletons

    @lru_cache(CacheMax)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        "Recursively check if a dependent type contains any resource dependency"
        if not (resolved_node := self._analyzed_nodes.get(dep_type)):
            resolved_node = self.analyze(dep_type)

        if self.is_registered_singleton(resolved_node.dependent):
            return False

        if resolved_node.is_resource:
            return True

        contain_resource = any(
            self.should_be_scoped(param.param_type)
            for _, param in resolved_node.dependencies.items()
        )
        return contain_resource

    def check_param_conflict(
        self, param_type: IDependent[T], current_path: list[IDependent[T]]
    ):
        if param_type in current_path:
            i = current_path.index(param_type)
            cycle = current_path[i:] + [param_type]
            raise CircularDependencyDetectedError(cycle)

        if (subnode := self._nodes.get(param_type)) and not subnode.config.reuse:
            if any(self._nodes[p].config.reuse for p in current_path):
                raise ReusabilityConflictError(current_path, param_type)

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
        Override a dependency within the current graph
        """

        self.remove_dependent(old_dep)
        self._node(new_dep)

    def search(self, dep_name: str) -> Union[DependentNode, None]:
        for dep, node in self._nodes.items():
            if dep.__name__ == dep_name:
                return node

    def analyze(
        self,
        dependent: INode[P, T],
        *,
        config: NodeConfig = DefaultConfig,
        ignore: Container[str] = EmptyIgnore,
    ) -> DependentNode:
        """
        Recursively analyze the type information of a dependency.
        """
        if node := self._analyzed_nodes.get(dependent):
            return node

        dependent_type = self._analyze_dep(dependent, config)

        if is_unsolvable_type(dependent_type):
            raise TopLevelBulitinTypeError(dependent_type)

        current_path: list[IDependent[T]] = []

        def dfs(dep: IDependent[T]) -> DependentNode:
            # when we register a concrete node we also register its bases to type_registry
            if dep in self._analyzed_nodes:
                return self._analyzed_nodes[dep]
            if dep not in self._type_registry:
                self._node(dep, config)

            node = self._resolve_concrete_node(dep)
            if self.is_registered_singleton(dep):
                return node

            current_path.append(dep)

            for param in node.analyze_unsolved_params(ignore):
                if (param_type := param.param_type) in self._analyzed_nodes:
                    continue
                self.check_param_conflict(param_type, current_path)
                if get_origin(param_type) is Annotated:
                    if use_meta := resolve_use(param_type):
                        ufunc, uconfig = use_meta
                        inode = self._node(ufunc, uconfig)
                        node.dependencies[param.name] = param.replace_type(
                            inode.dependent
                        )
                        self.analyze(inode.factory)
                        continue
                elif is_function(param_type):
                    fnode = DependentNode.from_node(param_type, config=config)
                    self._nodes[param_type] = fnode
                    node.dependencies[param.name] = param.replace_type(fnode.dependent)
                    self.analyze(fnode.factory)
                    continue
                if is_provided(param.default_):
                    continue
                if param.unresolvable:
                    raise UnsolvableDependencyError(
                        dep_name=param.name,
                        dependent_type=dep,
                        dependency_type=param.param_type,
                        factory=node.factory,
                    )
                try:
                    fnode = dfs(param_type)
                except UnsolvableNodeError as une:
                    une.add_context(node.dependent, param.name, param.param_type)
                    raise
                self._register_node(fnode)
                self._analyzed_nodes[param_type] = fnode
            current_path.pop()
            self._analyzed_nodes[dep] = node
            return node

        return dfs(dependent_type)

    def analyze_params(
        self, ufunc: Callable[P, T], config: NodeConfig = DefaultConfig
    ) -> tuple[bool, list[tuple[str, IDependent[Any]]]]:
        deps = Dependencies.from_signature(
            signature=get_typed_signature(ufunc), function=ufunc, config=config
        )
        depends_on_resource: bool = False
        unresolved: list[tuple[str, IDependent[Any]]] = []

        for name, dep in deps.items():
            param_type = dep.param_type

            if dep.unresolvable:
                continue

            if use_meta := (resolve_use(param_type) or resolve_use(dep.default_)):
                ufunc, uconfig = use_meta
                use_node = self._node(ufunc, uconfig)
                param_type = use_node.dependent

            self.analyze(param_type, config=config)
            depends_on_resource = depends_on_resource or self.should_be_scoped(
                param_type
            )
            unresolved.append((name, param_type))
        return depends_on_resource, unresolved

    def analyze_nodes(self) -> None:
        unsolved_nodes: set[IDependent[Any]] = (
            self._nodes.keys() - self._analyzed_nodes.keys()
        )
        for node_type in unsolved_nodes:
            self.analyze(node_type)

    def register_singleton(
        self, dependent: T, dependent_type: Union[type[T], None] = None
    ) -> None:
        """
        Register a dependent instance to be injected, provide a dependent type if it's not the same as the instance type.
        """
        if dependent_type is None:
            dependent_type = type(dependent)
        register_dependent(self._resolved_singletons, dependent_type, dependent)
        self._registered_singletons.add(dependent_type)

    def resolve_callback(
        self,
        resolved: Any,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T:
        if factory_type not in ("default", "function"):
            raise ResourceOutsideScopeError(dependent)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, resolved)
        return resolved

    async def aresolve_callback(
        self,
        resolved: Any,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ):
        if factory_type not in ("default", "function"):
            raise ResourceOutsideScopeError(dependent)

        instance = await resolved if isawaitable(resolved) else resolved
        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return cast(T, instance)

    @overload
    def resolve(
        self,
        dependent: IDependent[T],
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
        if args:
            raise PositionalOverrideError(args)

        if resolution := self._resolved_singletons.get(dependent):
            return resolution

        provided_params = frozenset(overrides)
        node: DependentNode = self.analyze(dependent, ignore=provided_params)
        return _resolve_dfs(
            self, self._nodes, self._resolved_singletons, node.dependent, overrides
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

        if resolution := self._resolved_singletons.get(dependent):
            return resolution

        provided_params = frozenset(overrides)
        node: DependentNode = self.analyze(dependent, ignore=provided_params)
        return await _aresolve_dfs(
            self, self._nodes, self._resolved_singletons, node.dependent, overrides
        )

    def _node(
        self, dependent: INode[P, T], config: NodeConfig = DefaultConfig
    ) -> DependentNode:
        merged_config = NodeConfig(
            reuse=config.reuse, ignore=self._ignore | config.ignore
        )
        node = DependentNode.from_node(dependent, config=merged_config)
        if node.function_dependent:
            return node

        if ori_node := self._nodes.get(node.dependent):
            if should_override(node, ori_node):
                self._remove_node(ori_node)
        self._register_node(node)
        return node

    @overload
    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: Literal[False] = False,
    ) -> "SyncScope": ...

    @overload
    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: Literal[True],
    ) -> "AsyncScope": ...

    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: bool = False,
    ) -> Union["SyncScope", "AsyncScope"]:
        """
        Get most recently created scope

        as_async: bool
        pure typing helper, ignored at runtime
        """
        scope = _SCOPE_CONTEXT.get()
        if name and name != scope.name:
            return scope.get_scope(name)
        return scope

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

        config = NodeConfig(**iconfig)
        require_scope, unresolved = self.analyze_params(func, config=config)

        def replace(
            before: Maybe[IDependent[T]] = MISSING,
            after: Maybe[IDependent[T]] = MISSING,
            **kw_overrides: type[Any],
        ):
            nonlocal unresolved

            for i, (pname, ptype) in enumerate(unresolved):
                if ptype is before and is_provided(after):
                    analyzed = self.analyze(after).dependent
                    unresolved[i] = (pname, analyzed)

                if pname in kw_overrides:
                    analyzed = self.analyze(kw_overrides[pname]).dependent
                    unresolved[i] = (pname, analyzed)

        if iscoroutinefunction(func):
            if require_scope:

                @wraps(func)
                async def _async_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    async with self.ascope() as scope:
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
            sync_func = cast(IDependent[T], func)
            if require_scope:

                @wraps(sync_func)
                def _sync_scoped_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                    with self.scope() as scope:
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
        dg = Graph()

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

    def add_nodes(
        self, *nodes: Union[IDependent[T], tuple[IDependent[T], INodeConfig]]
    ) -> None:
        for node in nodes:
            if isinstance(node, tuple):
                node, nconfig = node
                self.node(node, **nconfig)
            else:
                self.node(node)


cdef class ResolveScope(Resolver):
    __slots__ = ()

    _name: "Maybe[Hashable]"
    _pre: "Maybe[ResolveScope]"
    _stack: "Union[ExitStack, AsyncExitStack]"

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def pre(self) -> Maybe[ResolveScope]:
        return self._pre

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name or MISSING}, "
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_singletons)})"
        )

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


cdef class SyncScope(ResolveScope):
    __slots__ = ScopeSlots

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        self._name = name
        self._pre = pre
        self._stack = ExitStack()

        super().__init__(
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )

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
        resolved: ContextManager[T],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ):
        if factory_type in ("default", "function"):
            if is_reuse:
                register_dependent(self._resolved_singletons, dependent, resolved)
            return resolved

        if factory_type == "aresource":
            raise AsyncResourceInSyncError(dependent)
        instance = self.enter_context(resolved)
        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance


cdef class AsyncScope(ResolveScope):
    __slots__ = AsyncScopeSlots

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        self._name = name
        self._pre = pre
        self._stack = AsyncExitStack()
        self._loop = get_running_loop()

        super().__init__(
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )

    async def __aenter__(self) -> "AsyncScope":
        return self

    async def __aexit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        await self._stack.__aexit__(exc_type, exc_value, traceback)

    async def enter_async_context(self, context: AsyncContextManager[T]) -> T:
        return await self._stack.enter_async_context(context)

    @overload
    async def resolve(self, dependent: IDependent[T], /) -> T: ...

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
        resolved: Union[ContextManager[T], AsyncContextManager[T]],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T:
        if factory_type in ("default", "function"):
            instance = await resolved if isawaitable(resolved) else resolved
            if is_reuse:
                register_dependent(self._resolved_singletons, dependent, instance)
            return cast(T, instance)

        if factory_type == "resource":
            resolved = cast(ContextManager[T], resolved)
            resolved = syncscope_in_thread(self._loop, self._workers, resolved)
        else:
            resolved = cast(AsyncContextManager[T], resolved)

        instance = await self.enter_async_context(resolved)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance


cdef class Graph(Resolver):
    """
    ### Description:
    A Directed Acyclic Graph where each dependent is a node.

    Example
    ---
    ```python
    from typing import AsyncGenerator
    from ididi import Graph, use, entry, AsyncResource

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

    __slots__ = GraphSlots

    _nodes: GraphNodes[Any]
    "Map a type to a dependent node"
    _analyzed_nodes: dict[IDependent[Any], DependentNode]
    "Nodes that have been recursively resolved and is validated to be resolvable."
    _type_registry: TypeRegistry
    "Map a type to its implementations"
    _resolved_singletons: ResolvedSingletons[Any]
    "Instances of reusable types"
    _registered_singletons: set[type]
    "Types that are menually added singletons "

    def __init__(
        self,
        *,
        self_inject: bool = True,
        ignore: GraphIgnoreConfig = EmptyIgnore,
        workers: Union[ThreadPoolExecutor, None] = None,
    ):
        """
        - self_inject:

          - If True, the current instance of the `Graph` will be injected into any dependency that requires it.

        - ignore:

          - A configuration that defines dependencies to be ignored during resolution.
          - This can be a tuple of `(dependency_name, dependency_type)`.
          - For example, `ignore=('name', str)` will ignore dependencies named `'name'` or those of type `str`.


        ### Example:
        - Creating a Graph with self-injection enabled, ignoring 'str' type dependencies, and allowing partial resolution.

            ```python
            graph = Graph(self_inject=False, ignore=('name', str))
            ```

        ### Notes:
        - `self_inject` is useful when you have a dependency that requires current instance of `Graph` as its dependency.
        - `ignore` is a flexible configuration to skip over specific dependencies during resolution.
        """
        config = GraphConfig(self_inject=self_inject, ignore=ignore)

        super().__init__(
            nodes=dict(),
            analyzed_nodes=dict(),
            type_registry=TypeRegistry(),
            resolved_singletons=dict(),
            registered_singletons=set(),
            ignore=config.ignore,
            workers=workers or ThreadPoolExecutor(thread_name_prefix="ididi"),
        )

        if self_inject:
            self.register_singleton(self)

        default_scope = self._create_scope(DefaultScopeName)
        self._token = _SCOPE_CONTEXT.set(default_scope)

    def _merge_nodes(self, other: "Graph"):
        for dep_type, other_node in other.nodes.items():
            current_node = self._nodes.get(dep_type)
            current_resolved = self._analyzed_nodes.get(dep_type)
            other_resolved = other.resolved_nodes.get(dep_type)

            if not current_node or (should_override(other_node, current_node)):
                self._nodes[dep_type] = other_node
                if other_resolved:
                    self._analyzed_nodes[dep_type] = other_resolved
                continue

            # the conflict case, we only update if current node is not resolved
            if not current_resolved and other_resolved:
                self._analyzed_nodes[dep_type] = other_resolved

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
            other_scope = other.use_scope()
            if other_scope.name is not DefaultScopeName:
                raise MergeWithScopeStartedError()

            self._merge_nodes(other)
            self._type_registry.update(other._type_registry)
            self._resolved_singletons.update(other._resolved_singletons)

    def reset(self, clear_nodes: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_nodes: bool
        ---
        whether to clear:
        - the node registry
        - the type registry
        """
        self._resolved_singletons.clear()
        self._registered_singletons.clear()

        if clear_nodes:
            self._type_registry.clear()
            self._analyzed_nodes.clear()
            self._nodes.clear()
