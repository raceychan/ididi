from functools import partial
from typing import Any, Awaitable, Callable, Union, cast, overload

from typing_extensions import Unpack

from .graph import Graph as Graph
from .interfaces import INodeConfig, TEntryDecor
from .utils.typing_utils import P, T


@overload
def entry(**iconfig: Unpack[INodeConfig]) -> TEntryDecor: ...


@overload
def entry(func: Callable[P, T]) -> Callable[..., T]: ...


def entry(
    func: Union[Callable[P, T], None] = None,
    **iconfig: Unpack[INodeConfig],
) -> Union[Callable[..., Union[T, Awaitable[T]]], TEntryDecor]:
    if not func:
        return cast(TEntryDecor, partial(entry, **iconfig))
    return Graph().entry(func, **iconfig)


@overload
def resolve(dep: Callable[P, T], /) -> T: ...


@overload
def resolve(dep: Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve(dep: Callable[P, T], /, **overrides: Any) -> T:
    return Graph().resolve(dep, **overrides)
