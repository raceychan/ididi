import inspect
import typing as ty
from dataclasses import dataclass

EMPTY_SIGNATURE = inspect.Signature()
INSPECT_EMPTY = inspect.Signature.empty

type IFactory[I, **P] = ty.Callable[P, I]


# class Override[T](ty.Protocol):
#     @ty.overload
#     def __call__[**P](self, dep: ty.Callable[P, T], /) -> T: ...

#     @ty.overload
#     def __call__[
#         **P
#     ](self, dep: ty.Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T: ...

#     def __call__[
#         **P
#     ](self, dep: ty.Callable[P, T], /, *args: ty.Any, **kwargs: ty.Any) -> T: ...


class TDecor:
    @ty.overload
    def __call__[I](self, factory: type[I]) -> type[I]: ...

    @ty.overload
    def __call__[I, **P](self, factory: IFactory[I, P]) -> IFactory[I, P]: ...

    def __call__[
        I, **P
    ](self, factory: IFactory[I, P] | type[I]) -> IFactory[I, P] | type[I]: ...


class INodeConfig(ty.TypedDict, total=False):
    """
    reuse: bool
    ---
    whether the resolved instanec should be reused if it already exists in the graph.
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
