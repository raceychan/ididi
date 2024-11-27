from typing import Any, Awaitable, Callable, Union, overload

from .graph import DependencyGraph as DependencyGraph
from .utils.typing_utils import P, T


def entry(func: Callable[P, T]) -> Union[Callable[..., T], Callable[..., Awaitable[T]]]:
    dg = DependencyGraph()
    return dg.entry(func)


@overload
def resolve(dep: Callable[P, T], /) -> T: ...


@overload
def resolve(dep: Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve(dep: Callable[P, T], /, **overrides: Any) -> T:
    dg = DependencyGraph()
    return dg.resolve(dep, **overrides)
