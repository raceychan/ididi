from typing import Callable, Literal, Union

from typing_extensions import TypeGuard

from .typing_utils import T


class _Missed:
    """
    Sentinel object to represent a null value.
    bool(NULL) is False.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> Literal[False]:
        return False


MISSING = _Missed()


Maybe = Union[T, _Missed]
"""
Nullable[int] == int | NULL
"""


def is_provided(value: Maybe[T]) -> TypeGuard[T]:
    """
    Check if the value is not MISSING.
    """
    return value is not MISSING


def use_var(
    obj_type: type[T],
) -> tuple[Callable[[], Maybe[T]], Callable[[T], None]]:
    instance: Maybe[T] = MISSING

    def getter() -> Maybe[T]:
        return instance

    def setter(obj: T) -> None:
        if not isinstance(obj, obj_type):
            raise TypeError
        nonlocal instance
        instance = obj

    return getter, setter
