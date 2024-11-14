import typing as ty

from .graph import DependencyGraph as DependencyGraph


def entry[
    R, **P
](func: ty.Callable[P, R]) -> ty.Callable[..., R] | ty.Callable[..., ty.Awaitable[R]]:
    dg = DependencyGraph()
    return dg.entry(func)


@ty.overload
def resolve[**P, T](dep: ty.Callable[P, T], /) -> T: ...


@ty.overload
def resolve[
    T, **P
](dep: ty.Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve[**P, T](dep: ty.Callable[P, T], /, **overrides: ty.Any) -> T:
    dg = DependencyGraph()
    return dg.resolve(dep, **overrides)
