import typing as ty
from dataclasses import dataclass

type IFactory[I, **P] = ty.Callable[P, I]


class TDecor:
    """
    NOTE: can't switch the order of factory: type[I] and ty.Callable[P, I]
    because type[I] -> type[I] is a subtype of ty.Callable[P, I] -> ty.Callable[P, I]
    so that type[I] would be ignored by the overload resolution
    """

    @ty.overload
    def __call__[I](self, factory: type[I]) -> type[I]: ...

    @ty.overload
    def __call__[I, **P](self, factory: IFactory[I, P]) -> IFactory[I, P]: ...

    def __call__[
        I, **P
    ](self, factory: IFactory[I, P] | type[I]) -> IFactory[I, P] | type[I]: ...


class INodeConfig(ty.TypedDict, total=False):
    """
    Only used for type hinting, typing.Unpack[INodeConfig]
    For actual config, use NodeConfig
    """

    reuse: bool
    # lazy: bool


@dataclass(kw_only=True, frozen=True, slots=True, unsafe_hash=True)
class NodeConfig:
    reuse: bool = True
