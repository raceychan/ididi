import typing as ty

__all__ = ["NULL", "Nullable", "is_not_null"]


class _Null:
    """
    Sentinel object to represent a null value.
    bool(NULL) is False.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "NULL"

    def __bool__(self) -> bool:
        return False


NULL = _Null()


type Nullable[T] = T | _Null
"""
Nullable[int] == int | NULL
"""


def is_not_null[T](value: Nullable[T]) -> ty.TypeGuard[T]:
    """
    Check if the value is not NULL.
    """
    return value is not NULL
