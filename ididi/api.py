from functools import partial
from typing import Any, Awaitable, Callable, Union, cast, overload

from typing_extensions import Unpack

from ._itypes import IAsyncFactory, IEntryConfig, IFactory, TEntryDecor
from .graph import DependencyGraph as DependencyGraph
from .utils.typing_utils import P, T


@overload
def entry(**iconfig: Unpack[IEntryConfig]) -> TEntryDecor: ...


@overload
def entry(func: IFactory[P, T]) -> Callable[..., T]: ...


def entry(
    func: Union[IFactory[P, T], IAsyncFactory[P, T], None] = None,
    **iconfig: Unpack[IEntryConfig],
) -> Union[Callable[..., Union[T, Awaitable[T]]], TEntryDecor]:
    if not func:
        return cast(TEntryDecor, partial(entry, **iconfig))
    return DependencyGraph().entry(func, **iconfig)


@overload
def resolve(dep: Callable[P, T], /) -> T: ...


@overload
def resolve(dep: Callable[P, T], /, *args: P.args, **overrides: P.kwargs) -> T: ...


def resolve(dep: Callable[P, T], /, **overrides: Any) -> T:
    return DependencyGraph().resolve(dep, **overrides)
