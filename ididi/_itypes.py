import inspect
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Coroutine,
    Generator,
    Protocol,
    TypedDict,
    Union,
    overload,
    runtime_checkable,
)

from .utils.typing_utils import P, R, T

EMPTY_SIGNATURE = inspect.Signature()
INSPECT_EMPTY = inspect.Signature.empty

IEmptyFactory = Callable[[], R]
IEmptyAsyncFactory = Callable[[], Awaitable[R]]

IFactory = Callable[P, R]
IAnyFactory = Callable[..., R]
IAsyncFactory = Callable[P, Awaitable[R]]
IAnyAsyncFactory = Callable[..., Awaitable[R]]
IResourceFactory = IFactory[P, Generator[R, None, None]]
IAsyncResourceFactory = Callable[P, AsyncGenerator[R, None]]

INode = Union[
    IFactory[P, R],
    IAnyFactory[R],
    IAsyncFactory[P, R],
    IAnyAsyncFactory[R],
    IResourceFactory[P, R],
    IAsyncResourceFactory[P, R],
    type[R],
]


class TDecor(Protocol):
    @overload
    def __call__(self, factory: type[T]) -> type[T]: ...

    @overload
    def __call__(self, factory: IResourceFactory[P, T]) -> IResourceFactory[P, T]: ...

    @overload
    def __call__(
        self, factory: IAsyncResourceFactory[P, T]
    ) -> IAsyncResourceFactory[P, T]: ...

    @overload
    def __call__(self, factory: IAsyncFactory[P, T]) -> IAsyncFactory[P, T]: ...

    @overload
    def __call__(self, factory: IFactory[P, T]) -> IFactory[P, T]: ...

    @overload
    def __call__(self, factory: INode[P, T]) -> INode[P, T]: ...


class INodeConfig(TypedDict, total=False):
    """
    reuse: bool
    ---
    whether the resolved instance should be reused if it already exists in the graph.

    lazy: bool
    ---
    whether the resolved instance should be a `lazy` dependent, meaning that its dependencies would not be resolved untill the attribute is accessed.

    partial
    ---
    whether to ignore bulitin types when statically resolve

    ignore
    ---
    types or names to ignore
    """

    reuse: bool
    lazy: bool
    partial: bool
    ignore: tuple[Union[str, type], ...]


class NodeConfig:
    __slots__ = ("reuse", "lazy", "partial", "ignore")

    ignore: tuple[Union[str, type], ...]

    def __init__(
        self,
        *,
        reuse: bool = True,
        lazy: bool = False,
        partial: bool = False,
        ignore: Union[tuple[Union[str, type], ...], None] = None,
    ):
        self.reuse = reuse
        self.lazy = lazy
        self.partial = partial
        self.ignore = ignore or ()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.reuse=}, {self.lazy=}, {self.partial=}, {self.ignore=})"


class GraphConfig:
    __slots__ = "static_resolve"

    def __init__(self, *, static_resolve: bool = True):
        self.static_resolve = static_resolve


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class AsyncClosable(Protocol):
    async def close(self) -> Coroutine[Any, Any, None]: ...
