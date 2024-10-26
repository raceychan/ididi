import typing as ty

__all__ = ["NULL", "Nullable", "is_not_null"]


class _Null:
    pass

    def __repr__(self) -> str:
        return "NULL"


NULL = _Null()


type Nullable[T] = T | _Null


def is_not_null[T](value: Nullable[T]) -> ty.TypeGuard[T]:
    return not isinstance(value, _Null)
