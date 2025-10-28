from functools import partial
from typing import Any, Awaitable, Callable, Union, cast, overload

from .graph import Graph as Graph
from .interfaces import IDependent, NodeIgnoreConfig, TEntryDecor, Maybe, MISSING
from .utils.typing_utils import P, T


@overload
def entry(*, reuse: Maybe[bool] = MISSING, ignore: NodeIgnoreConfig = ()) -> TEntryDecor: ...


@overload
def entry(func: Callable[P, T], /) -> IDependent[T]: ...

def entry(
    func: Union[Callable[P, T], None] = None,
    /,
    *,
    reuse: Maybe[bool] = MISSING, 
    ignore: NodeIgnoreConfig = (),
) -> Union[Callable[..., Union[T, Awaitable[T]]], TEntryDecor]:
    if not func:
        return cast(TEntryDecor, partial(entry, reuse=reuse, ignore=ignore))
    return Graph().entry(func, reuse=reuse, ignore=ignore)


@overload
def resolve(dep: Callable[P, T], /) -> T: ...


@overload
def resolve(dep: Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve(dep: Callable[P, T], /, **overrides: Any) -> T:
    return Graph().resolve(dep, **overrides)
