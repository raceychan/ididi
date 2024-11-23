import typing as ty

import typing_extensions as tyex

from .typing_utils import T


class _Missed:
    """
    Sentinel object to represent a null value.
    bool(NULL) is False.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> ty.Literal[False]:
        return False


MISSING = _Missed()


Maybe = ty.Union[T, _Missed]
"""
Nullable[int] == int | NULL
"""


def is_provided(value: Maybe[T]) -> tyex.TypeGuard[T]:
    """
    Check if the value is not MISSING.
    """
    return value is not MISSING


def use_var(
    obj_type: type[T],
) -> tuple[ty.Callable[[], Maybe[T]], ty.Callable[[T], None]]:
    instance: Maybe[T] = MISSING

    def getter() -> Maybe[T]:
        return instance

    def setter(obj: T) -> None:
        if not isinstance(obj, obj_type):
            raise TypeError
        nonlocal instance
        instance = obj

    return getter, setter
