import typing as ty

# __all__ = ["NULL", "Nullable", "is_not_null"]


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


MISSING_TYPE = _Missed
MISSING = _Missed()


type Maybe[T] = T | _Missed
"""
Nullable[int] == int | NULL
"""


def is_provided[T](value: Maybe[T]) -> ty.TypeGuard[T]:
    """
    Check if the value is not MISSING.
    """
    return not isinstance(value, _Missed)
