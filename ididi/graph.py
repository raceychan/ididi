from asyncio import AbstractEventLoop, get_running_loop
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token, copy_context
from functools import lru_cache, partial, wraps
from inspect import isawaitable, iscoroutinefunction
from types import MappingProxyType, MethodType, TracebackType
from typing import (
    Annotated,
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    ContextManager,
    Final,
    Hashable,
    Literal,
    Sequence,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Self, Unpack

from ._ds import GraphNodes, GraphNodesView, ResolvedSingletons, TypeRegistry, Visitor
from ._node import DependentNode, resolve_meta, should_override
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
    ConfigConflictError,
    DeprecatedError,
    MergeWithScopeStartedError,
    NotSupportedError,
    OutOfScopeError,
    ParamReusabilityConflictError,
    PositionalOverrideError,
    ResourceOutsideScopeError,
    ReusabilityConflictError,
    TopLevelBulitinTypeError,
    UnsolvableNodeError,
)
from .interfaces import (  # INodeConfig,
    EntryFunc,
    GraphIgnore,
    GraphIgnoreConfig,
    IAsyncFactory,
    IDependent,
    IFactory,
    INode,
    NodeIgnore,
    NodeIgnoreConfig,
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
    mapping: dict[IDependent[Any], Any],
    dependent_type: IDependent[T],
    instance: T,
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
    nodes: GraphNodes,
    cache: dict[IDependent[T], T],
    ptype: IDependent[T],
    overrides: dict[str, Any],
) -> T:

    if resolution := cache.get(ptype):
        return resolution

    params = {}
    pnode = nodes.get(ptype) or resolver.analyze(ptype)

    for param in pnode.dependencies:
        name = param.name
        if (val := overrides.get(name, MISSING)) is not MISSING:
            params[name] = val
        elif param.should_be_ignored:
            continue
        else:
            params[name] = _resolve_dfs(
                resolver, nodes, cache, param.param_type, overrides
            )

    try:
        instance = pnode.factory(**params)
    except TypeError as te:
        raise TypeError(f"{pnode.dependent}, {te}")

    result = resolver.resolve_callback(
        instance,
        pnode.dependent,
        pnode.factory_type,
        pnode.reuse,
    )
    return result


async def _aresolve_dfs(
    resolver: "Resolver",
    nodes: GraphNodes,
    cache: dict[IDependent[T], T],
    ptype: IDependent[T],
    overrides: dict[str, Any],
) -> T:
    if resolution := cache.get(ptype):
        return resolution

    params = {}
    pnode = nodes.get(ptype) or resolver.analyze(ptype)

    for param in pnode.dependencies:
        name = param.name
        if (val := overrides.get(name, MISSING)) is not MISSING:
            params[name] = val
        elif param.should_be_ignored:
            continue
        else:
            params[name] = await _aresolve_dfs(
                resolver, nodes, cache, param.param_type, overrides
            )

    instance = pnode.factory(**params)
    resolved = await resolver.aresolve_callback(
        instance,
        pnode.dependent,
        pnode.factory_type,
        pnode.reuse,
    )
    return cast(T, resolved)


@asynccontextmanager
async def syncscope_in_thread(
    loop: AbstractEventLoop, workers: ThreadPoolExecutor, cm: ContextManager[T]
):
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
    analyzed_nodes: GraphNodes
    type_registry: TypeRegistry
    ignore: GraphIgnore
    workers: ThreadPoolExecutor


class Resolver:

    def __init__(
        self,
        name: Maybe[Hashable],
        resolved_singletons: ResolvedSingletons,
        registered_singletons: set[IDependent[Any]],
        **args: Unpack[SharedData],
    ):
        self._name = name
        self._nodes = args["nodes"]
        self._analyzed_nodes = args["analyzed_nodes"]
        self._type_registry = args["type_registry"]
        self._ignore = args["ignore"]
        self._workers = args["workers"]

        self._registered_singletons = registered_singletons
        self._resolved_singletons = resolved_singletons

    @property
    def name(self) -> Hashable:
        return self._name

    def __repr__(self) -> str:
        name = f"name={self._name!r}, " if self._name is not MISSING else ""
        return (
            f"{self.__class__.__name__}("
            f"{name}"
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_singletons)})"
        )

    def get(self, dep: IDependent[Any]) -> Union[DependentNode, None]:
        if node := self._nodes.get(dep):
            return node
        return self._nodes.get(resolve_node_type(dep))

    @property
    def nodes(self) -> GraphNodesView:
        return MappingProxyType(self._nodes)

    @property
    def resolution_registry(self) -> ResolvedSingletons:
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
        dependent = get_origin(dep_type) or dep_type
        if dependent in self._nodes:
            return

        self._nodes[dependent] = node
        self._type_registry.register(dependent)

    def _resolve_concrete_node(self, dependent: IDependent[Any]) -> DependentNode:
        # when user assign impl via factory
        if (node := self._nodes.get(dependent)) and node.factory_type != "default":
            return node

        concrete_type = self._type_registry[dependent][-1]
        concrete_node = self._nodes[concrete_type]
        concrete_node.check_for_implementations()
        return concrete_node

    def _analyze_dep(self, dependent: INode[P, T], reuse: Maybe[bool], ignore: NodeIgnoreConfig) -> INode[P, T]:
        if isinstance(dependent, MethodType):
            dep_method = dependent
            bound_obj = dep_method.__self__
            if not isinstance(bound_obj, type):
                raise NotSupportedError("Instance method is not supported")
            self.include_node(dep_method, reuse, ignore)
            dependent = bound_obj
            resolved_type = resolve_annotation(dependent)
        elif is_function(dependent):
            if is_new_type(dependent):
                resolved_type = resolve_annotation(dependent)
            else:
                if dependent not in self._nodes:
                    self.include_node(dependent, reuse, ignore)
                resolved_type = resolve_factory(dependent)
        else:
            resolved_type = resolve_annotation(dependent)


        if get_origin(resolved_type) is Annotated:
            if (node_meta := resolve_meta(resolved_type)) and node_meta.factory:
                self.include_node(dependent=node_meta.factory, reuse=node_meta.reuse)
            elif is_function(dependent):
                if node := self._nodes.get(dependent):
                    return dependent
                node = DependentNode.from_node(dependent, reuse=reuse, ignore=ignore)
                self._nodes[dependent] = node
                return dependent

            resolved_type, *_ = get_args(resolved_type)
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
    def should_be_scoped(self, dep_type: IDependent[Any]) -> bool:
        if (meta := resolve_meta(dep_type)) and meta.ignore:
            return False
        if not (resolved_node := self._analyzed_nodes.get(dep_type)):
            try:
                resolved_node = self.analyze(dep_type)
            except TopLevelBulitinTypeError as tle:
                raise UnsolvableNodeError(f"Failed to parse dependency {dep_type}, {tle.message}") from tle

        if self.is_registered_singleton(resolved_node.dependent):
            return False

        if resolved_node.is_resource:
            return True

        ignore_params = resolved_node.ignore
        for param in resolved_node.dependencies:
            pname = param.name
            ptype = param.param_type

            if pname in ignore_params or ptype in ignore_params:
                continue

            if param.should_be_ignored:
                continue

            try:
                if self.should_be_scoped(ptype):
                    return True
            except UnsolvableNodeError as une:
                une.add_context(resolved_node.dependent, pname, ptype)
                raise une

        return False

    def check_param_conflict(
        self, param_type: IDependent[T], current_path: list[IDependent[T]]
    ):
        if param_type in current_path:
            i = current_path.index(param_type)
            cycle = current_path[i:] + [param_type]
            raise CircularDependencyDetectedError(cycle)

        if (subnode := self._nodes.get(param_type)) and not subnode.reuse:
            if any(self._nodes[p].reuse for p in current_path):
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
        dependent: IDependent[Any],
        *,
        reuse: Maybe[bool] = MISSING,
        ignore: NodeIgnore = EmptyIgnore,
    ) -> DependentNode:
        if node := self._analyzed_nodes.get(dependent):
            return node

        dependent_type = self._analyze_dep(dependent, reuse, ignore)

        if is_unsolvable_type(dependent_type):
            raise TopLevelBulitinTypeError(dependent_type)

        current_path: list[IDependent[Any]] = []

        def dfs(dep: IDependent[Any]) -> DependentNode:
            # when we register a concrete node we also register its bases to type_registry
            node_graph_ignore: NodeIgnoreConfig

            if dep in self._analyzed_nodes:
                return self._analyzed_nodes[dep]
            if dep not in self._type_registry:
                self.include_node(dep, reuse=reuse, ignore=ignore)

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
                if (node_meta := resolve_meta(param_type)) and node_meta.factory:
                    try:
                        inode = self.include_node(dependent=node_meta.factory, reuse=node_meta.reuse)
                    except ConfigConflictError as cce:
                        raise ParamReusabilityConflictError(node.factory.__qualname__, param.name, cce)

                    node.update_param(
                        param.name, inode.dependent
                    )
                    self.analyze(inode.factory, ignore=node_graph_ignore)
                    continue
                elif is_function(param_type):
                    fnode = DependentNode.from_node(param_type, reuse=reuse, ignore=ignore)
                    self._nodes[fnode.dependent] = fnode
                    node.update_param(param.name, fnode.dependent)
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
        self, ufunc: IFactory[P, T], reuse: Maybe[bool], ignore: NodeIgnore,
    ) -> tuple[bool, list[tuple[str, IDependent[Any]]]]:
        """
        Used solely in `entry`
        """
        sig = get_typed_signature(ufunc)
        node = DependentNode.create(dependent=ufunc, factory=ufunc, factory_type="function", signature=sig, reuse=reuse, ignore=ignore)

        depends_on_resource: bool = False
        unresolved: list[tuple[str, IDependent[Any]]] = []

        for i, param in enumerate(node.dependencies):
            name = param.name
            param_type = param.param_type
            if not (node_meta := resolve_meta(param.param_type)) or node_meta.ignore:
                continue

            if resolve_meta(param.default_):
                raise DeprecatedError(f"Using default value {param} for `use` is not longer supported")

            try:
                use_node = self.include_node(dependent=node_meta.factory, reuse=node_meta.reuse)
            except ConfigConflictError as cce:
                raise ParamReusabilityConflictError(node.factory.__qualname__, param.name, cce)

            param_type = use_node.dependent
            if any(x in ignore for x in (i, name, param_type)):
                continue

            self.analyze(param_type, reuse=reuse, ignore=ignore)
            depends_on_resource = depends_on_resource or self.should_be_scoped(
                param_type
            )
            unresolved.append((name, param_type))
        return depends_on_resource, unresolved

    def analyze_nodes(self) -> None:
        unsolved_node_types: set[IDependent[Any]] = (
            self._nodes.keys() - self._analyzed_nodes.keys()
        )
        for node_type in unsolved_node_types:
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
            factory_type="default",
            reuse=True,
            ignore=EmptyIgnore,
        )

        self._type_registry[dependent_type] = [dependent_type]

    def resolve_callback(
        self,
        resolved: T,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: Maybe[bool],
    ) -> T:
        if factory_type in ("resource", "aresource"):
            raise ResourceOutsideScopeError(dependent)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, resolved)
        return resolved

    async def aresolve_callback(
        self,
        resolved: Any,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: Maybe[bool],
    ) -> T:
        if factory_type in ("resource", "aresource"):
            raise ResourceOutsideScopeError(dependent)

        instance = await resolved if isawaitable(resolved) else resolved
        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance

    @overload
    def resolve(
        self,
        dependent: INode[P, T],
        /,
    ) -> T: ...

    @overload
    def resolve(
        self,
        dependent: INode[P, T],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    def resolve(
        self,
        dependent: INode[P, T],
        /,
        *args: Any,
        **overrides: Any,
    ) -> T:
        if args:
            raise PositionalOverrideError(args)

        if resolution := self._resolved_singletons.get(dependent):
            return resolution

        provided_params = tuple(overrides)
        node: DependentNode = self.analyze(dependent, ignore=provided_params)

        return _resolve_dfs(
            self, self._nodes, self._resolved_singletons, node.dependent, overrides
        )
    @overload
    async def aresolve(
        self,
        dependent: INode[P, T],
        /,
    ) -> T: ...

    @overload
    async def aresolve(
        self,
        dependent: INode[P, T],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    async def aresolve(
        self,
        dependent: INode[P, T],
        /,
        *args: Any,
        **overrides: Any,
    ) -> T:
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
        self, dependent: INode[P, T], reuse: Maybe[bool] = MISSING, ignore: NodeIgnoreConfig=EmptyIgnore
    ) -> DependentNode:
        merged_ignore: NodeIgnoreConfig = self._ignore + ignore # type: ignore
        node = DependentNode.from_node(dependent, reuse=reuse, ignore=merged_ignore)
        if ori_node := self._nodes.get(node.dependent):
            if not should_override(node, ori_node):
                ori_node.update_reusability(node)
                return node
            if not ori_node.dependent in self._registered_singletons:
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
        scope = _SCOPE_CONTEXT.get()
        if name and name != scope.name:
            return scope.get_scope(name)
        return scope

    @overload
    def entry(self, *, reuse: Maybe[bool]=MISSING, ignore: NodeIgnoreConfig=EmptyIgnore) -> TEntryDecor: ...
    @overload
    def entry(
        self, func: IFactory[P, T], *, reuse: Maybe[bool]=MISSING, ignore: NodeIgnoreConfig=EmptyIgnore
    ) -> EntryFunc[P, T]: ...

    def entry(
        self,
        func: Union[IFactory[P, T], IAsyncFactory[P, T], None] = None,
        *,
        reuse: Maybe[bool] = MISSING,
        ignore: NodeIgnoreConfig = EmptyIgnore
    ) -> Union[EntryFunc[P, T], TEntryDecor]:
        if not func:
            configured = partial(self.entry, reuse=reuse, ignore=ignore)
            return cast(EntryFunc[P, T], configured)

        if not isinstance(ignore, tuple):
            ignore = (ignore, )
        require_scope, unresolved = self.analyze_params(func, reuse=reuse, ignore=ignore)

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
            sync_func = cast(IFactory[P, T], func)
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


    @overload
    def node(self, dependent: type[T], *,  reuse: Maybe[bool]=MISSING, ignore: NodeIgnoreConfig=EmptyIgnore) -> type[T]: ...


    @overload
    def node(self, dependent: Callable[P, T], *,  reuse: Maybe[bool]=MISSING, ignore: NodeIgnoreConfig=EmptyIgnore) -> Callable[P, T]: ...

    @overload
    def node(self, *, reuse: Maybe[bool]=MISSING, ignore: NodeIgnoreConfig=EmptyIgnore) -> Callable[[T], T]: ...

    def node(
        self,
        dependent:  Union[INode[P, T], None] = None,
        *,
        reuse: Maybe[bool] = MISSING, 
        ignore: NodeIgnoreConfig = EmptyIgnore,
    ) -> Union[INode[P, T], TDecor]:
        if not dependent:
            configured = partial(self.node, reuse=reuse, ignore=ignore)
            return configured  # type: ignore

        if get_origin(dependent) is Annotated:
            if node_meta := resolve_meta(dependent) :
                self.include_node(dependent=node_meta.factory, reuse=node_meta.reuse)
                return dependent
            else:
                dependent = get_args(dependent)[0]
                
        if is_unsolvable_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        if not isinstance(ignore, tuple):
            ignore = (ignore, )
        self.include_node(dependent, reuse=reuse, ignore=ignore)
        return dependent

    def add_nodes(
        self, *nodes: Any
    ) -> None:
        for node in nodes:
            if isinstance(node, tuple):
                raise DeprecatedError(
                        "Passing (node, config) tuples to Graph.add_nodes is no longer supported,"
                        "Use idid.use instead, e.g. "
                        "add_nodes(use(Service, reuse=True))."
                )
            self.node(node)


class ResolveScope(Resolver):
    _name: "Maybe[Hashable]"
    _pre: "Maybe[ResolveScope]"

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



    def get_scope(self, name: Hashable) -> "Self":
        if name == self._name:
            return self

        ptr = self
        while is_provided(ptr._pre):
            if ptr._pre._name == name:
                return ptr._pre # type: ignore
            ptr = ptr._pre
        else:
            raise OutOfScopeError(name)

    def register_exit_callback(self, callback: IDependent[None]) -> None:
        raise NotImplementedError


class SyncScope(ResolveScope):
    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons,
        registered_singletons: set[IDependent[Any]],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        super().__init__(
            name=name,
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )

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

    def enter_context(self, context: ContextManager[T]) -> T:
        return self._stack.enter_context(context)

    def resolve_callback(
        self,
        resolved: ContextManager[T],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: Maybe[bool],
    ) -> T:
        if factory_type in ("default", "function"):
            if is_reuse:
                register_dependent(self._resolved_singletons, dependent, resolved)
            return cast(T, resolved)

        if factory_type == "aresource":
            raise AsyncResourceInSyncError(dependent)
        instance = self.enter_context(resolved)
        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance

    def register_exit_callback(
        self, cb: IFactory[P, None], *args: P.args, **kwargs: P.kwargs
    ):
        self._stack.callback(cb, *args, **kwargs)


class AsyncScope(ResolveScope):
    _stack: AsyncExitStack

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons,
        registered_singletons: set[IDependent[Any]],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union["SyncScope", "AsyncScope"]] = MISSING,
        **args: Unpack[SharedData],
    ):
        super().__init__(
            name=name,
            **args,
            resolved_singletons=resolved_singletons,
            registered_singletons=registered_singletons,
        )
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

    def enter_context(self, context: ContextManager[T]) -> T:
        return self._stack.enter_context(context)

    async def enter_async_context(
        self, context: Union[ContextManager[T], AsyncContextManager[T]]
    ) -> T:
        return await self._stack.enter_async_context(context)

    @overload
    async def resolve(
        self,
        dependent: INode[P, T],
        /,
    ) -> T: ...

    @overload
    async def resolve(
        self,
        dependent: INode[P, T],
        /,
        *args: P.args,
        **overrides: P.kwargs,
    ) -> T: ...

    async def resolve(
        self, dependent: INode[P, T], /, *args: Any, **kwargs:Any 
    ) -> T:
        return await super().aresolve(dependent, *args, **kwargs)

    async def aresolve_callback(
        self,
        resolved: Union[ContextManager[T], AsyncContextManager[T]],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: Maybe[bool],
    ) -> T:
        if factory_type in ("default", "function", "afunction"):
            fac = cast(Union[T, Awaitable[T]], resolved)
            if isawaitable(fac):
                instance: T = await fac
            else:
                instance: T = fac
            if is_reuse:
                register_dependent(self._resolved_singletons, dependent, instance)
            return instance

        if factory_type == "resource":
            resolved = cast(ContextManager[T], resolved)
            wrapped  = syncscope_in_thread(self._loop, self._workers, resolved)
            resolved = cast(AsyncContextManager[T], wrapped)

        instance = await self.enter_async_context(resolved)

        if is_reuse:
            register_dependent(self._resolved_singletons, dependent, instance)
        return instance

    def register_exit_callback(
        self, cb: IFactory[P, Awaitable[None]], *args: P.args, **kwargs: P.kwargs
    ):
        self._stack.push_async_callback(cb, *args, **kwargs)


class Graph(Resolver):
    _nodes: GraphNodes
    _analyzed_nodes: dict[Hashable, DependentNode]
    _type_registry: TypeRegistry
    _resolved_singletons: ResolvedSingletons
    _registered_singletons: set[type]

    def __init__(
        self,
        name: Maybe[str] = MISSING,
        *,
        self_inject: bool = True,
        ignore: GraphIgnoreConfig = EmptyIgnore,
        workers: Union[ThreadPoolExecutor, None] = None,
    ):
        config = GraphConfig(self_inject=self_inject, ignore=ignore)

        super().__init__(
            name=name,
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

            current_node.update_reusability(other_node)

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
