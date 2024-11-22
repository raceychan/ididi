import typing as ty

from .graph import DependencyGraph as DependencyGraph
from .utils.typing_utils import P, T


def entry(
    func: ty.Callable[P, T]
) -> ty.Union[ty.Callable[..., T], ty.Callable[..., ty.Awaitable[T]]]:
    dg = DependencyGraph()
    return dg.entry(func)


@ty.overload
def resolve(dep: ty.Callable[P, T], /) -> T: ...


@ty.overload
def resolve(dep: ty.Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve(dep: ty.Callable[P, T], /, **overrides: ty.Any) -> T:
    dg = DependencyGraph()
    return dg.resolve(dep, **overrides)
