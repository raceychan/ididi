from typing import Any,  ForwardRef,  Mapping, TypeVar, Union
from typing import _eval_type as ty_eval_type  # type: ignore
from typing import cast

from typing_extensions import ParamSpec, TypeGuard

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")
C = TypeVar("C", covariant=True)



PrimitiveBuiltins = type[Union[int, float, complex, str, bool, bytes, bytearray]]
ContainerBuiltins = type[
    Union[list[T], tuple[T, ...], dict[Any, T], set[T], frozenset[T]]
]
BuiltinSingleton = type[None]


def is_builtin_primitive(t: Any) -> TypeGuard[PrimitiveBuiltins]:
    return t in {int, float, complex, str, bool, bytes, bytearray, type}


def is_builtin_container(t: Any) -> TypeGuard[ContainerBuiltins[Any]]:
    return t in {list, tuple, dict, set, frozenset}


def is_builtin_singleton(t: Any) -> TypeGuard[BuiltinSingleton]:
    return t is None or t is type(None)


def is_builtin_type(
    t: Any,
) -> TypeGuard[Union[PrimitiveBuiltins, ContainerBuiltins[Any], BuiltinSingleton]]:
    """
    Builtin types are ignored at type resolving.
    It must be provided by default value, or use a factory to override it.
    typing.Any is also ignored at type resolving.
    """

    # TODO: we might name this as 'is_unresolved_type' as corner cases increase.
    is_primitive = is_builtin_primitive(t)
    is_container = is_builtin_container(t)
    is_singleton = is_builtin_singleton(t)
    return is_primitive or is_container or is_singleton or (t is Any)


def eval_type(
    value: ForwardRef,
    globalns: Union[dict[str, Any], None] = None,
    localns: Union[Mapping[str, Any], None] = None,
    *,
    lenient: bool = False,
) -> Any:
    """
    # NOTE: copy from pydantic, credit to them.
    Evaluate the annotation using the provided namespaces.

    Args:
        value: The value to evaluate. If `None`, it will be replaced by `type[None]`. If an instance
            of `str`, it will be converted to a `ForwardRef`.
        localns: The global namespace to use during annotation evaluation.
        globalns: The local namespace to use during annotation evaluation.
        lenient: Whether to keep unresolvable annotations as is or re-raise the `NameError` exception. Default: re-raise.
    """

    try:
        return cast(type[Any], ty_eval_type(value, globalns, localns))
    except NameError:
        if not lenient:
            raise
        return value


def actualize_strforward(annotation: Any, gvars: dict[str, Any]) -> Any:
    """
    solve cases where user refer to a existing class using a string.
    e.g.
    ```py
    class UserService: ...

    def user_service_factory() -> "UserService": ...
    ```
    Here "UserService" is a string not `typing.ForwardRef`
    """

    if isinstance(annotation, str):
        annotation = ForwardRef(annotation, is_argument=False)
        annotation = eval_type(annotation, gvars, gvars, lenient=True)
    return annotation
