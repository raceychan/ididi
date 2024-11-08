import typing as ty

from .graph import DependencyGraph as DependencyGraph
from .types import INodeConfig, Override


def entry[
    R, **P
](
    func: ty.Callable[P, R],
    /,
    **kwargs: ty.Unpack[INodeConfig],
) -> (
    ty.Callable[..., R] | ty.Callable[..., ty.Awaitable[R]]
):
    dg = DependencyGraph()
    return dg.entry_node(func, **kwargs)



def solve[
    T, **P
](cls: ty.Callable[P, T], /, *p_args: P.args, **p_kwargs: P.kwargs) -> Override[T]:
    dg = DependencyGraph()
    return dg.resolve(cls, *p_args, **p_kwargs)
