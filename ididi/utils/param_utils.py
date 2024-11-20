import typing as ty


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


type Maybe[T] = T | _Missed
"""
Nullable[int] == int | NULL
"""


def is_provided[T](value: Maybe[T]) -> ty.TypeGuard[T]:
    """
    Check if the value is not MISSING.
    """
    return value is not MISSING


def use_var[
    T
](obj_type: type[T]) -> tuple[ty.Callable[[], Maybe[T]], ty.Callable[[T], None]]:
    instance: Maybe[T] = MISSING

    def getter() -> Maybe[T]:
        return instance

    def setter(obj: T) -> None:
        if not isinstance(obj, obj_type):
            raise TypeError
        nonlocal instance
        instance = obj

    return getter, setter
