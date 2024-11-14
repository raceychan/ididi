import inspect
import typing as ty
from dataclasses import dataclass

EMPTY_SIGNATURE = inspect.Signature()
INSPECT_EMPTY = inspect.Signature.empty

type IFactory[**P, R] = ty.Callable[P, R]
type IAsyncFactory[**P, R] = ty.Callable[P, ty.Awaitable[R]]


@ty.runtime_checkable
class Closable(ty.Protocol):
    def close(self) -> None: ...


@ty.runtime_checkable
class AsyncClosable(ty.Protocol):
    async def close(self) -> ty.Coroutine[ty.Any, ty.Any, None]: ...


class TDecor(ty.Protocol):
    @ty.overload
    def __call__[I](self, factory: type[I]) -> type[I]: ...

    @ty.overload
    def __call__[**P, *I](self, factory: IFactory[P, I]) -> IFactory[P, I]: ...

    def __call__[
        **P, I
    ](self, factory: IFactory[P, I] | type[I]) -> IFactory[P, I] | type[I]: ...


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


@dataclass(kw_only=True, frozen=True, slots=True, unsafe_hash=True)
class NodeConfig:
    reuse: bool = True
    lazy: bool = False


@dataclass(kw_only=True, frozen=True, slots=True, unsafe_hash=True)
class GraphConfig:
    static_resolve: bool = True
