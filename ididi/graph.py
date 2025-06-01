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
    ContextManager,
    Final,
    Hashable,
    Sequence,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import Self, TypeAliasType, Unpack

from ._ds import GraphNodes, GraphNodesView, ResolvedSingletons, TypeRegistry, Visitor
from ._node import (
    IGNORE_PARAM_MARK,
    DefaultConfig,
    Dependencies,
    Dependency,
    DependentNode,
    NodeConfig,
    resolve_use,
    search_meta,
    should_override,
)
from ._type_resolve import (
    flatten_annotated,
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
    UnsolvableNodeError,
)
from .interfaces import (
    EntryFunc,
    GraphIgnore,
    GraphIgnoreConfig,
    IAsyncFactory,
    IDependent,
    IFactory,
    INode,
    INodeConfig,
    TDecor,
    TEntryDecor,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

AnyScope = Union["SyncScope", "AsyncScope"]
ScopeToken = Token[AnyScope]
ScopeContext = ContextVar[AnyScope]
_SCOPE_CONTEXT: Final[ScopeContext] = ContextVar("idid_scope_ctx")


def register_dependent(
    mapping: dict[Hashable, Any],
    dependent_type: object,
    instance: object,
):
    if isinstance(dependent_type, type):
        base_types = get_bases(type(instance))
        for base in base_types:
            if base in mapping:
                continue
            mapping[base] = instance

    mapping[dependent_type] = instance


def _resolve_dfs(
    resolver: "Resolver",
    nodes: dict[type, DependentNode],
    cache: dict[type, Any],
    ptype: type,
    overrides: dict[str, Any],
):

    if resolution := cache.get(ptype):
        return resolution

    params = {}
    pnode = nodes.get(ptype) or resolver.analyze(ptype)

    for name, param in pnode.dependencies.items():
        if (val := overrides.get(name, MISSING)) is not MISSING:
            params[name] = val
        elif param.should_be_ignored:
            continue
        else:
            params[name] = _resolve_dfs(resolver, nodes, cache, param.param_type, overrides)  # type: ignore

    try:
        instance = pnode.factory(**params)
    except TypeError as te:
        raise TypeError(f"{pnode.dependent}, {te}")

    result = resolver.resolve_callback(
        instance,
        pnode.dependent,
        pnode.factory_type,
        pnode.config.reuse,
    )
    return result


async def _aresolve_dfs(
    resolver: "Resolver",
    nodes: dict[type, Any],
    cache: dict[type, Any],
    ptype: type[T],
    overrides: dict[str, Any],
) -> T:
    if resolution := cache.get(ptype):
        return resolution

    params = {}
    pnode = nodes.get(ptype) or resolver.analyze(ptype)

    for name, param in pnode.dependencies.items():
        if (val := overrides.get(name, MISSING)) is not MISSING:
            params[name] = val
        elif param.should_be_ignored:
            continue
        else:
            params[name] = await _aresolve_dfs(resolver, nodes, cache, param.param_type, overrides)  # type: ignore

    instance = pnode.factory(**params)
    resolved = await resolver.aresolve_callback(
        instance,
        pnode.dependent,
        pnode.factory_type,
        pnode.config.reuse,
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

    nodes: GraphNodes
    analyzed_nodes: dict[Hashable, DependentNode]
    type_registry: TypeRegistry
    ignore: GraphIgnore
    workers: ThreadPoolExecutor


class Resolver:
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
    def nodes(self) -> GraphNodesView:
        return MappingProxyType(self._nodes)

    @property
    def resolution_registry(self) -> ResolvedSingletons[Any]:
        return self._resolved_singletons

    @property
    def type_registry(self) -> TypeRegistry:
        return self._type_registry

    @property
    def resolved_nodes(self) -> GraphNodesView:
        return MappingProxyType(self._analyzed_nodes)

    @property
    def visitor(self) -> Visitor:
        return Visitor(self._nodes)

    def __contains__(self, item: INode[P, T]) -> bool:
        return resolve_node_type(item) in self._nodes

    def _remove_node(self, node: DependentNode) -> None:
        dependent_type = node.dependent

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolved_singletons.pop(dependent_type, None)
        self._analyzed_nodes.pop(dependent_type, None)

    def _register_node(self, node: DependentNode) -> None:
        dep_type = node.dependent
        dependent: Hashable = get_origin(dep_type) or dep_type
        if dependent in self._nodes:
            return

        self._nodes[dependent] = node
        self._type_registry.register(dependent)

    def _resolve_concrete_node(self, dependent: Hashable) -> DependentNode:
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
            self.include_node(dep_method)
            dependent = bound_obj
            resolved_type = resolve_annotation(dependent)
        elif is_function(dependent):
            if is_new_type(dependent):
                resolved_type = resolve_annotation(dependent)
            else:
                if dependent not in self._nodes:
                    self.include_node(dependent, config)
                resolved_type = resolve_factory(dependent)
        else:
            resolved_type = resolve_annotation(dependent)

        annt_dep = None

        if get_origin(resolved_type) is Annotated:
            annt_dep = resolved_type

        if annt_dep:
            if use_meta := resolve_use(annt_dep):
                ufunc, uconfig = use_meta
                self.include_node(ufunc, uconfig)

            if is_function(dependent):
                # if dependent in nodes just reuse
                if node := self._nodes.get(dependent):
                    return dependent

                node = DependentNode.from_node(dependent, config=config)
                self._nodes[dependent] = node
                return dependent

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

    def is_registered_singleton(self, dependent_type: Hashable) -> bool:
        return dependent_type in self._registered_singletons

    @lru_cache(CacheMax)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        if get_origin(dep_type) is Annotated:
            metas = flatten_annotated(dep_type)
            if IGNORE_PARAM_MARK in metas:
                return False
        if not (resolved_node := self._analyzed_nodes.get(dep_type)):
            resolved_node = self.analyze(dep_type)

        if self.is_registered_singleton(resolved_node.dependent):
            return False

        if resolved_node.is_resource:
            return True

        ignore_params = resolved_node.config.ignore
        for pname, param in resolved_node.dependencies.items():
            ptype = param.param_type

            if pname in ignore_params or ptype in ignore_params:
                continue

            if param.should_be_ignored:
                continue

            if self.should_be_scoped(ptype):
                return True
        return False

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
        if dependent_type in self._registered_singletons:
            self._registered_singletons.remove(dependent_type)

    def remove_dependent(self, dependent: INode[P, T]) -> None:
        dependent_type = resolve_node_type(dependent)

        if node := self._nodes.get(dependent_type):
            self._remove_node(node)

    def override(self, old_dep: INode[P, T], new_dep: INode[P, T]) -> None:
        self.remove_dependent(old_dep)
        self.include_node(new_dep)

    def search(self, dep_name: str) -> Union[DependentNode, None]:
        for dep, node in self._nodes.items():
            if dep.__name__ == dep_name:
                return node

    def analyze(
        self,
        dependent: INode[P, T],
        *,
        config: NodeConfig = DefaultConfig,
        ignore: tuple[str] = EmptyIgnore,
    ) -> DependentNode:
        if node := self._analyzed_nodes.get(dependent):
            return node

        dependent_type = self._analyze_dep(dependent, config)

        if is_unsolvable_type(dependent_type):
            raise TopLevelBulitinTypeError(dependent_type)

        current_path: list[IDependent[T]] = []

        def dfs(dep: Hashable) -> DependentNode:
            # when we register a concrete node we also register its bases to type_registry
            node_graph_ignore: tuple[Union[str, type, TypeAliasType], ...]

            if dep in self._analyzed_nodes:
                return self._analyzed_nodes[dep]
            if dep not in self._type_registry:
                self.include_node(dep, config)

            node = self._resolve_concrete_node(dep)
            if self.is_registered_singleton(dep):
                return node

            current_path.append(dep)

            if ignore is not self._ignore:
                node_graph_ignore = ignore + self._ignore
            else:
                node_graph_ignore = ignore

            for param in node.analyze_unsolved_params(node_graph_ignore):
                if (param_type := param.param_type) in self._analyzed_nodes:
                    continue
                self.check_param_conflict(param_type, current_path)
                if get_origin(param_type) is Annotated:
                    metas = flatten_annotated(param_type)
                    if use_meta := search_meta(metas):
                        ufunc, uconfig = use_meta
                        inode = self.include_node(ufunc, uconfig)
                        node.dependencies[param.name] = param.replace_type(
                            inode.dependent
                        )
                        self.analyze(inode.factory, ignore=node_graph_ignore)
                        continue
                elif is_function(param_type):
                    fnode = DependentNode.from_node(param_type, config=config)
                    self._nodes[param_type] = fnode
                    node.dependencies[param.name] = param.replace_type(fnode.dependent)
                    self.analyze(fnode.factory, ignore=node_graph_ignore)
                    continue

                if is_provided(param.default_) or param.should_be_ignored:
                    continue

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
            signature=get_typed_signature(ufunc), function=ufunc
        )
        depends_on_resource: bool = False
        unresolved: list[tuple[str, IDependent[Any]]] = []

        for i, (name, param) in enumerate(deps.items()):
            param_type = param.param_type

            if (
                i in config.ignore
                or name in config.ignore
                or param_type in config.ignore
            ):
                continue
            if is_provided(param.default_) or param.should_be_ignored:
                continue

            if get_origin(param_type) is Annotated:
                metas = flatten_annotated(param_type)
                if use_meta := search_meta(metas):
                    ufunc, uconfig = use_meta
                    use_node = self.include_node(ufunc, uconfig)
                    param_type = use_node.dependent
                elif IGNORE_PARAM_MARK in metas:
                    continue
            elif use_meta := resolve_use(param.default_):
                ufunc, uconfig = use_meta
                use_node = self.include_node(ufunc, uconfig)
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
        if dependent_type is None:
            dependent_type = type(dependent)

        register_dependent(self._resolved_singletons, dependent_type, dependent)
        self._registered_singletons.add(dependent_type)

        self._nodes[dependent_type] = DependentNode(
            dependent=dependent_type,
            factory=dependent_type,
            factory_type="function",
            dependencies=Dependencies(),
            config=DefaultConfig,
        )

        self._type_registry[dependent_type] = [dependent_type]

    def resolve_callback(
        self,
        resolved: object,
        dependent: object,
        factory_type: str,
        is_reuse: bool,
    ):
        if factory_type in ("resource", "aresource"):
            raise ResourceOutsideScopeError(dependent)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, resolved)
        return resolved

    async def aresolve_callback(
        self,
        resolved: object,
        dependent: object,
        factory_type: str,
        is_reuse: bool,
    ):
        if factory_type in ("resource", "aresource"):
            raise ResourceOutsideScopeError(dependent)

        instance = await resolved if isawaitable(resolved) else resolved
        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance

    def resolve(
        self,
        dependent,
        /,
        *args,
        **overrides,
    ):
        if args:
            raise PositionalOverrideError(args)

        if resolution := self._resolved_singletons.get(dependent):
            return resolution

        provided_params = tuple(overrides)
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
    ):
        if args:
            raise PositionalOverrideError(args)

        if resolution := self._resolved_singletons.get(dependent):
            return resolution

        provided_params = tuple(overrides)
        node: DependentNode = self.analyze(dependent, ignore=provided_params)
        return await _aresolve_dfs(
            self, self._nodes, self._resolved_singletons, node.dependent, overrides
        )

    def include_node(
        self, dependent: INode[P, T], config: NodeConfig = DefaultConfig
    ) -> DependentNode:
        merged_config = NodeConfig(
            reuse=config.reuse, ignore=self._ignore + config.ignore
        )

        node = DependentNode.from_node(dependent, config=merged_config)
        if ori_node := self._nodes.get(node.dependent):
            if should_override(node, ori_node):
                self._remove_node(ori_node)
        self._register_node(node)
        return node

    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: bool = False,
    ) -> Union["SyncScope", "AsyncScope"]:
        scope = _SCOPE_CONTEXT.get()
        if name and name != scope.name:
            return scope.get_scope(name)
        return scope

    def entry(
        self,
        func: Union[Callable[P, T], IAsyncFactory[P, T], None] = None,
        **iconfig: Unpack[INodeConfig],
    ) -> Union[EntryFunc[P, T], TEntryDecor]:
        if not func:
            configured = partial(self.entry, **iconfig)
            return cast(EntryFunc[P, T], configured)

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
            sync_func = func
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
        return f

    def node(
        self,
        dependent: Union[INode[P, T], None] = None,
        **config: Unpack[INodeConfig],
    ) -> Union[INode[P, T], TDecor]:
        if not dependent:
            configured = partial(self.node, **config)
            return configured  # type: ignore

        if is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        self.include_node(dependent, config=NodeConfig(**config))
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


class ResolveScope(Resolver):
    _name: "Maybe[Hashable]"
    _pre: "Maybe[ResolveScope]"
    _stack: "Union[ExitStack, AsyncExitStack]"

    @property
    def name(self) -> Hashable:
        return self._name

    @property
    def pre(self) -> Maybe["ResolveScope"]:
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
                return ptr._pre
            ptr = ptr._pre
        else:
            raise OutOfScopeError(name)

    def register_exit_callback(self, callback: Callable[..., None]) -> None:
        raise NotImplementedError


class SyncScope(ResolveScope):
    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        super().__init__(
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )

        self._name = name
        self._pre = pre
        self._stack = ExitStack()

    def __enter__(self) -> "SyncScope":
        return self

    def __exit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None:
        self._stack.__exit__(exc_type, exc_value, traceback)

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

    def register_exit_callback(self, cb: Callable[P, None], *args, **kwargs):
        self._stack.callback(cb, *args, **kwargs)


class AsyncScope(ResolveScope):
    _stack: AsyncExitStack

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        super().__init__(
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )
        self._name = name
        self._pre = pre
        self._stack = AsyncExitStack()
        self._loop = get_running_loop()

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

    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        return await super().aresolve(dependent, *args, **kwargs)

    async def aresolve_callback(
        self,
        resolved: Union[ContextManager[T], AsyncContextManager[T]],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T:
        if factory_type in ("default", "function", "afunction"):
            instance = await resolved if isawaitable(resolved) else resolved
            if is_reuse:
                register_dependent(self._resolved_singletons, dependent, instance)
            return instance

        if factory_type == "resource":
            resolved = syncscope_in_thread(self._loop, self._workers, resolved)

        instance = await self.enter_async_context(resolved)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance

    def register_exit_callback(self, cb: Callable[P, None], *args, **kwargs):
        self._stack.push_async_callback(cb, *args, **kwargs)


class Graph(Resolver):
    _nodes: GraphNodes
    _analyzed_nodes: dict[Hashable, DependentNode]
    _type_registry: TypeRegistry
    _resolved_singletons: ResolvedSingletons[Any]
    _registered_singletons: set[type]

    def __init__(
        self,
        *,
        self_inject: bool = True,
        ignore: GraphIgnoreConfig = EmptyIgnore,
        workers: Union[ThreadPoolExecutor, None] = None,
    ):
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
        self._resolved_singletons.clear()
        self._registered_singletons.clear()

        if clear_nodes:
            self._type_registry.clear()
            self._analyzed_nodes.clear()
            self._nodes.clear()
