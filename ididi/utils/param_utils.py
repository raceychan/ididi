import typing as ty


class _Missed:
    """
    Sentinel object to represent a null value.
    bool(NULL) is False.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missed()


type Maybe[T] = T | _Missed
"""
Nullable[int] == int | NULL
"""


def is_provided[T](value: Maybe[T]) -> ty.TypeGuard[T]:
    """
    Check if the value is not MISSING.
    """
    return value is not MISSING
