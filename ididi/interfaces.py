from inspect import Signature
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Generator,
    Protocol,
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
INode = Union[
    INodeFactory[P, R], type[R], TypeAliasType, GenericAlias, Annotated[R, "Any"]
]

NodeIgnore = tuple[Union[str, int, type, TypeAliasType], ...]
GraphIgnore = tuple[Union[str, type, TypeAliasType], ...]

NodeConfigParam = Union[str, int, type, TypeAliasType]
GraphConfigParam = Union[str, type, TypeAliasType]

NodeIgnoreConfig = Union[NodeConfigParam, NodeIgnore]
GraphIgnoreConfig = Union[GraphConfigParam, GraphIgnore]

Resource = Generator[T, None, None]
AsyncResource = AsyncGenerator[T, None]


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


# class INodeConfig(TypedDict, total=False):
#     """
#     reuse: bool
#     ---
#     whether the resolved instance should be reused if it already exists in the graph.

#     partial
#     ---
#     whether to ignore bulitin types when statically resolve

#     ignore
#     ---
#     types or names to ignore
#     """

#     ignore: NodeIgnoreConfig
#     reuse: bool


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
