import inspect
import typing as ty

from .utils.typing_utils import P, R, T

EMPTY_SIGNATURE = inspect.Signature()
INSPECT_EMPTY = inspect.Signature.empty


IFactory = ty.Callable[P, R]
IAnyFactory = ty.Callable[..., R]
IAsyncFactory = ty.Callable[P, ty.Awaitable[R]]
IAnyAsyncFactory = ty.Callable[..., ty.Awaitable[R]]
IResourceFactory = ty.Callable[P, ty.Generator[R, None, None]]
IAsyncResourceFactory = ty.Callable[P, ty.AsyncGenerator[R, None]]
INode = ty.Union[
    IFactory[P, R],
    IAnyFactory[R],
    IAsyncFactory[P, R],
    IAnyAsyncFactory[R],
    IResourceFactory[P, R],
    IAsyncResourceFactory[P, R],
    type[R],
]


class TDecor(ty.Protocol):
    @ty.overload
    def __call__(self, factory: type[T]) -> type[T]: ...

    @ty.overload
    def __call__(self, factory: IResourceFactory[P, T]) -> IResourceFactory[P, T]: ...

    @ty.overload
    def __call__(
        self, factory: IAsyncResourceFactory[P, T]
    ) -> IAsyncResourceFactory[P, T]: ...

    @ty.overload
    def __call__(self, factory: IAsyncFactory[P, T]) -> IAsyncFactory[P, T]: ...

    @ty.overload
    def __call__(self, factory: IFactory[P, T]) -> IFactory[P, T]: ...

    @ty.overload
    def __call__(self, factory: INode[P, T]) -> INode[P, T]: ...


class INodeConfig(ty.TypedDict, total=False):
    """
    reuse: bool
    ---
    whether the resolved instance should be reused if it already exists in the graph.

    lazy: bool
    ---
    whether the resolved instance should be a `lazy` dependent, meaning that its dependencies would not be resolved untill the attribute is accessed.
    """

    reuse: bool
    lazy: bool
    partial: bool


class NodeConfig:
    __slots__ = ("reuse", "lazy", "partial")

    def __init__(
        self, *, reuse: bool = True, lazy: bool = False, partial: bool = False
    ):
        self.reuse = reuse
        self.lazy = lazy
        self.partial = partial


class GraphConfig:
    __slots__ = "static_resolve"

    def __init__(self, *, static_resolve: bool = True):
        self.static_resolve = static_resolve


@ty.runtime_checkable
class Closable(ty.Protocol):
    def close(self) -> None: ...


@ty.runtime_checkable
class AsyncClosable(ty.Protocol):
    async def close(self) -> ty.Coroutine[ty.Any, ty.Any, None]: ...
