from inspect import Signature
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Protocol,
    TypedDict,
    Union,
    overload,
)

from typing_extensions import TypeAliasType

from .utils.param_utils import MISSING, Maybe
from .utils.typing_utils import C, P, R, T

EMPTY_SIGNATURE = Signature()
INSPECT_EMPTY = Signature.empty

IDependent = Callable[..., T]
IEmptyFactory = Callable[[], R]
IFactory = Callable[P, R]
IAsyncFactory = Callable[P, Awaitable[R]]
IResourceFactory = IFactory[P, Generator[R, None, None]]
IAsyncResourceFactory = Callable[P, AsyncGenerator[R, None]]

INodeAnyFactory = Union[
    IDependent[R],
    IDependent[Awaitable[R]],
    IDependent[Generator[R, None, None]],
    IDependent[AsyncGenerator[R, None]],
]
INodeFactory = Union[
    IFactory[P, R],
    IAsyncFactory[P, R],
    IResourceFactory[P, R],
    IAsyncResourceFactory[P, R],
]
INode = Union[INodeFactory[P, R], type[R]]

NodeIgnore = tuple[Union[str, int, type, TypeAliasType]]
GraphIgnore = tuple[Union[str, type, TypeAliasType]]

NodeConfigParam = Union[str, int, type, TypeAliasType]
GraphConfigParam = Union[str, type, TypeAliasType]

NodeIgnoreConfig = Union[NodeConfigParam, Iterable[NodeConfigParam]]
GraphIgnoreConfig = Union[GraphConfigParam, Iterable[GraphConfigParam]]

Resource = Generator[T, None, None]
AsyncResource = AsyncGenerator[T, None]

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

    partial
    ---
    whether to ignore bulitin types when statically resolve

    ignore
    ---
    types or names to ignore
    """

    ignore: NodeIgnoreConfig
    reuse: bool


# class Closable(Protocol):
#     def close(self) -> None: ...


# class AsyncClosable(Protocol):
#     async def close(self) -> Coroutine[Any, Any, None]: ...


class EntryFunc(Protocol[P, C]):
    __name__: str

    def __call__(self, *args: Any, **kwargs: Any) -> C: ...

    @overload
    def replace(
        self,
        before: type[T],
        after: type[T],
        **params: type[Any],
    ) -> None: ...

    @overload
    def replace(
        self,
        before: Maybe[type[T]] = MISSING,
        after: Maybe[type[T]] = MISSING,
        **params: type[Any],
    ) -> None: ...

    def replace(
        self,
        before: Maybe[type[T]] = MISSING,
        after: Maybe[type[T]] = MISSING,
        **params: type[Any],
    ) -> None: ...


class TEntryDecor(Protocol):
    def __call__(self, func: Callable[P, T]) -> EntryFunc[P, T]: ...
