from __future__ import annotations

from asyncio import AbstractEventLoop
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from contextvars import ContextVar, Token
from functools import lru_cache
from types import TracebackType
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    Container,
    ContextManager,
    Generator,
    Generic,
    Hashable,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    Union,
    final,
    overload,
)

from typing_extensions import Self, Unpack, override

from ._ds import GraphNodes, GraphNodesView, ResolvedSingletons, TypeRegistry, Visitor
from ._node import DefaultConfig, DependentNode, NodeConfig
from .config import CacheMax, EmptyIgnore, FactoryType
from .interfaces import (
    EntryFunc,
    GraphIgnore,
    GraphIgnoreConfig,
    IAsyncFactory,
    IDependent,
    IFactory,
    INode,
    INodeConfig,
    INodeFactory,
    TDecor,
    TEntryDecor,
)
from .utils.param_utils import MISSING, Maybe
from .utils.typing_utils import P, T

def resolve_dfs(
    resolver: "Resolver",
    nodes: GraphNodes,
    cache: ResolvedSingletons[Any],
    ptype: IDependent[T],
    overrides: dict[str, Any],
) -> T: ...
async def aresolve_dfs(
    graph: "Resolver",
    nodes: GraphNodes,
    cache: ResolvedSingletons[Any],
    ptype: IDependent[T],
    overrides: dict[str, Any],
) -> T: ...
@asynccontextmanager
async def syncscope_in_thread(
    loop: AbstractEventLoop, workers: ThreadPoolExecutor, cm: ContextManager[T]
) -> AsyncIterator[T]:
    yield  # type: ignore

class SharedData(TypedDict):
    "Data shared between graph and scope"

    nodes: GraphNodes
    analyzed_nodes: dict[IDependent[Any], DependentNode]
    type_registry: TypeRegistry
    ignore: GraphIgnore
    workers: ThreadPoolExecutor

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
AnyScope = Union[SyncScope, AsyncScope]
ScopeToken = Token[AnyScope]
ScopeContext = ContextVar[AnyScope]

class Resolver:
    __slots__ = SharedSlots

    def __init__(
        self,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        **args: Unpack[SharedData],
    ) -> None: ...
    def __repr__(self) -> str: ...
    def get(self, dep: INode[P, T]) -> Union[DependentNode, None]: ...
    @property
    def nodes(self) -> GraphNodesView: ...
    @property
    def resolution_registry(self) -> ResolvedSingletons[Any]: ...
    @property
    def type_registry(self) -> TypeRegistry:
        """
        A mapping of abstract types to their implementations.
        """
        ...

    @property
    def resolved_nodes(self) -> GraphNodesView:
        """
        A node is considered resolved if all its dependencies have been resolved.
        Which means all its forward dependencies have been resolved.
        This would also include builtin types.
        """
        ...

    @property
    def visitor(self) -> Visitor: ...
    def __contains__(self, item: INode[P, T]) -> bool: ...
    def _remove_node(self, node: DependentNode) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        ...

    def _register_node(self, node: DependentNode) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """
        ...

    def _resolve_concrete_node(self, dependent: IDependent[T]) -> DependentNode:
        # when user assign impl via factory
        ...

    def _analyze_dep(
        self, dependent: INode[P, T], config: NodeConfig
    ) -> IDependent[T]: ...
    def _create_scope(
        self, name: Hashable = "", pre: Maybe[AnyScope] = MISSING
    ) -> SyncScope: ...
    def _create_ascope(
        self, name: Hashable = "", pre: Maybe[AnyScope] = MISSING
    ) -> AsyncScope: ...
    @contextmanager
    def scope(self, name: Hashable = "") -> Generator[SyncScope, None, None]: ...
    @asynccontextmanager
    async def ascope(self, name: Hashable = "") -> AsyncIterator[AsyncScope]:
        yield  # type: ignore
    # =================  Public =================

    def is_registered_singleton(self, dependent_type: IDependent[T]) -> bool: ...
    @lru_cache(CacheMax)
    def should_be_scoped(self, dep_type: INode[P, T]) -> bool:
        "Recursively check if a dependent type contains any resource dependency"
        ...

    def check_param_conflict(
        self, param_type: IDependent[T], current_path: list[IDependent[T]]
    ) -> None: ...
    def remove_singleton(self, dependent_type: type) -> None:
        "Remove the registered singleton from current graph, return if not found"
        ...

    def remove_dependent(self, dependent: INode[P, T]) -> None:
        "Remove the dependent from current graph, return if not found"
        ...

    def override(self, old_dep: INode[P, T], new_dep: INode[P, T]) -> None:
        """
        globally override a dependency
        """
        ...

    def search(self, dep_name: str) -> Union[DependentNode, None]: ...
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
        ...

    def analyze_params(
        self, func: Callable[P, T], config: NodeConfig = DefaultConfig
    ) -> tuple[bool, list[tuple[str, IDependent[Any]]]]: ...
    def analyze_nodes(self) -> None: ...
    def register_singleton(
        self, dependent: T, dependent_type: Union[type[T], None] = None
    ) -> None:
        """
        Register a dependent instance to be injected, provide a dependent type if it's not the same as the instance type.
        """
        ...

    def resolve_callback(
        self,
        resolved: Any,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T: ...
    async def aresolve_callback(
        self,
        resolved: Any,
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T: ...
    def resolve(
        self,
        dependent: INode[..., T],
        /,
        *args: Any,
        **overrides: Any,
    ) -> T: ...
    async def aresolve(
        self,
        dependent: INode[..., T],
        /,
        *args: Any,
        **overrides: Any,
    ) -> T:
        """
        Async version of resolve that handles async context managers and coroutines.
        """

    def include_node(
        self, dependent: INode[P, T], config: NodeConfig = DefaultConfig
    ) -> DependentNode: ...
    @overload
    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: Literal[False] = False,
    ) -> SyncScope: ...
    @overload
    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: Literal[True],
    ) -> AsyncScope: ...
    def use_scope(
        self,
        name: Hashable = "",
        *,
        as_async: bool = False,
    ) -> Union[SyncScope, AsyncScope]:
        """
        Get most recently created scope

        as_async: bool
        pure typing helper, ignored at runtime
        """

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
        ...

    def add_nodes(
        self, *nodes: Union[IDependent[T], tuple[IDependent[T], INodeConfig]]
    ) -> None: ...

class ScopeMixin(Generic[Stack]):
    __slots__ = ()

    _nodes: GraphNodes
    _stack: Stack
    _name: Hashable
    _pre: Maybe[AnyScope]
    _workers: ThreadPoolExecutor
    _resolved_singletons: ResolvedSingletons[Any]
    _registered_singletons: set[type]

    @property
    def name(self) -> Hashable: ...
    @property
    def pre(self) -> Maybe[AnyScope]: ...
    def __repr__(self) -> str: ...
    def enter_context(self, context: ContextManager[T]) -> T: ...
    def get_scope(self, name: Hashable) -> Self: ...

class SyncScope(ScopeMixin[ExitStack], Resolver):
    __slots__ = ScopeSlots

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union[SyncScope, AsyncScope]] = MISSING,
        **args: Unpack[SharedData],
    ) -> None: ...
    def __enter__(self) -> SyncScope: ...
    def __exit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None: ...
    @override
    def resolve_callback(
        self,
        resolved: ContextManager[T],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T: ...
    def register_exit_callback(
        self, cb: Callable[P, None], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...

class AsyncScope(ScopeMixin[AsyncExitStack], Resolver):
    __slots__ = AsyncScopeSlots

    def __init__(
        self,
        *,
        resolved_singletons: ResolvedSingletons[Any],
        registered_singletons: set[type],
        name: Maybe[Hashable] = MISSING,
        pre: Maybe[Union[SyncScope, AsyncScope]] = MISSING,
        **args: Unpack[SharedData],
    ) -> None: ...
    async def __aenter__(self) -> AsyncScope: ...
    async def __aexit__(
        self,
        exc_type: Union[type[Exception], None],
        exc_value: Union[Exception, None],
        traceback: Union[TracebackType, None],
    ) -> None: ...
    async def enter_async_context(self, context: AsyncContextManager[T]) -> T: ...
    @overload
    async def resolve(self, dependent: IDependent[T], /) -> T: ...
    @overload
    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T: ...
    async def resolve(
        self, dependent: IFactory[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> T: ...
    @override
    async def aresolve_callback(
        self,
        resolved: Union[ContextManager[T], AsyncContextManager[T]],
        dependent: IDependent[T],
        factory_type: FactoryType,
        is_reuse: bool,
    ) -> T: ...
    def register_exit_callback(
        self, cb: Callable[P, Awaitable[None]], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...

@final
class Graph(Resolver):
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

    _nodes: GraphNodes
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
    ) -> None:
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
        ...

    def _merge_nodes(self, other: "Graph") -> None: ...

    # ==========================  Public ==========================

    def merge(self, other: Union["Graph", Sequence["Graph"]]) -> None:
        """
        Merge the other graphs into this graph, update the current graph.

        ## Logic

        1. If a node not in current graph, it will be added,
        2. otherwise, compare the `resolve order` of these two nodes, keep the one with higher order.

        ### resolve order

        1. node with resource management factory > node with a plain factory.
        2. node with a plain factory > node with default constructor.
        """
        ...

    def reset(self, clear_nodes: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_nodes: bool
        ---
        whether to clear:
        - the node registry
        - the type registry
        """
        ...
