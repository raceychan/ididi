import typing as ty
from dataclasses import dataclass

type T_Factory[I, **P] = ty.Callable[P, I] | type[I]
type TDecor[T_Factory] = ty.Callable[[T_Factory], T_Factory]


class INodeConfig(ty.TypedDict, total=False):
    reuse: bool

@dataclass(frozen=True)
class NodeConfig:
    reuse: bool = True



# ConfigDefault: INodeConfig = INodeConfig(reuse=True)


# def config_with_default(user_config: INodeConfig) -> INodeConfig:
#     return INodeConfig(**(ConfigDefault | user_config))
