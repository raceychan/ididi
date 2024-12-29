from inspect import Signature
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

from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, R, T

EMPTY_SIGNATURE = Signature()
INSPECT_EMPTY = Signature.empty

IEmptyFactory = Callable[[], R]

IFactory = Callable[P, R]
IAnyFactory = Callable[..., R]
IAsyncFactory = Callable[P, Awaitable[R]]
IAnyAsyncFactory = Callable[..., Awaitable[R]]
IResourceFactory = IFactory[P, Generator[R, None, None]]
IAsyncResourceFactory = Callable[P, AsyncGenerator[R, None]]

INodeAnyFactory = Union[
    Callable[..., R],
    Callable[..., Awaitable[R]],
    Callable[..., Generator[R, None, None]],
    Callable[..., AsyncGenerator[R, None]],
]
INodeFactory = Union[
    IFactory[P, R],
    IAsyncFactory[P, R],
    IResourceFactory[P, R],
    IAsyncResourceFactory[P, R],
]
INode = Union[INodeFactory[P, R], type[R]]

# P1 = TypeVar("P1")
# P2 = TypeVar("P2")
# P3 = TypeVar("P3")
# P4 = TypeVar("P4")
# P5 = TypeVar("P5")
# P6 = TypeVar("P6")
# P7 = TypeVar("P7")
# P8 = TypeVar("P8")
# P9 = TypeVar("P9")


# Factory with many type params
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


class TEntryDecor(Protocol):
    def __call__(self, func: Callable[P, T]) -> Callable[..., T]: ...


class IEntryConfig(TypedDict, total=False):
    lazy: bool
    ignore: Union[Union[str, type], tuple[Union[str, type], ...]]


class INodeConfig(IEntryConfig, total=False):
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


class NodeConfig:
    __slots__ = ("reuse", "lazy", "ignore")

    ignore: tuple[Union[str, type], ...]

    def __init__(
        self,
        *,
        reuse: bool = True,
        lazy: bool = False,
        ignore: Maybe[Union[Union[str, type], tuple[Union[str, type], ...]]] = MISSING,
    ):
        self.lazy = lazy

        if not is_provided(ignore):
            ignore = tuple()
        elif not isinstance(ignore, tuple):
            ignore = (ignore,)

        self.ignore = ignore
        self.reuse = reuse

    def __repr__(self):
        return f"{self.__class__.__name__}({self.reuse=}, {self.lazy=}, {self.ignore=})"


class EntryConfig(NodeConfig):
    __slots__ = ("lazy", "ignore", "reuse")

    def __init__(
        self,
        *,
        lazy: bool = False,
        ignore: Maybe[Union[Union[str, type], tuple[Union[str, type], ...]]] = MISSING,
    ):
        super().__init__(reuse=False, lazy=lazy, ignore=ignore)


class GraphConfig:
    __slots__ = "self_inject"

    def __init__(self, *, self_inject: bool = True):
        self.self_inject = self_inject


@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class AsyncClosable(Protocol):
    async def close(self) -> Coroutine[Any, Any, None]: ...
