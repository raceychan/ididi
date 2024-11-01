import typing as ty
from dataclasses import dataclass

type IFactory[I, **P] = ty.Callable[P, I] 
type TDecor[T_Factory] = ty.Callable[[T_Factory], T_Factory]


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
