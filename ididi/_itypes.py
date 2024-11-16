import inspect
import typing as ty
from dataclasses import dataclass

EMPTY_SIGNATURE = inspect.Signature()
INSPECT_EMPTY = inspect.Signature.empty

type IFactory[**P, R] = ty.Callable[P, R]
type IAsyncFactory[**P, R] = ty.Callable[P, ty.Awaitable[R]]
type IResourceFactory[**P, R] = ty.Callable[P, ty.Generator[R]]
type IAsyncResourceFactory[**P, R] = ty.Callable[P, ty.AsyncGenerator[R]]
type INode[**P, R] = IFactory[P, R] | IAsyncFactory[P, R] | IResourceFactory[
    P, R
] | IAsyncResourceFactory[P, R] | type[R]


class TDecor(ty.Protocol):
    @ty.overload
    def __call__[I](self, factory: type[I]) -> type[I]: ...

    @ty.overload
    def __call__[
        **P, I
    ](self, factory: IResourceFactory[P, I]) -> IResourceFactory[P, I]: ...

    @ty.overload
    def __call__[
        **P, I
    ](self, factory: IAsyncResourceFactory[P, I]) -> IAsyncResourceFactory[P, I]: ...

    @ty.overload
    def __call__[**P, I](self, factory: IAsyncFactory[P, I]) -> IAsyncFactory[P, I]: ...

    @ty.overload
    def __call__[**P, I](self, factory: IFactory[P, I]) -> IFactory[P, I]: ...

    @ty.overload
    def __call__[**P, I](self, factory: INode[P, I]) -> INode[P, I]: ...


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


@dataclass(kw_only=True, frozen=True, slots=True, unsafe_hash=True)
class NodeConfig:
    reuse: bool = True
    lazy: bool = False
    partial: bool = False


@dataclass(kw_only=True, frozen=True, slots=True, unsafe_hash=True)
class GraphConfig:
    static_resolve: bool = True


@ty.runtime_checkable
class Closable(ty.Protocol):
    def close(self) -> None: ...


@ty.runtime_checkable
class AsyncClosable(ty.Protocol):
    async def close(self) -> ty.Coroutine[ty.Any, ty.Any, None]: ...
