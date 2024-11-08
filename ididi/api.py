import typing as ty

from .graph import DependencyGraph as DependencyGraph
from .types import INodeConfig


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


@ty.overload
def solve[**P, T](dep: ty.Callable[P, T], /) -> T: ...


@ty.overload
def solve[
    T, **P
](dep: ty.Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def solve[**P, T](dep: ty.Callable[P, T], /, **overrides: ty.Any) -> T:
    dg = DependencyGraph()
    return dg.resolve(dep, **overrides)
