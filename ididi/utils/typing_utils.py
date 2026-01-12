import sys
from types import GenericAlias
from typing import Annotated, Any, ForwardRef, Hashable, Mapping, TypeVar, Union
from typing import _eval_type as ty_eval_type
from typing import cast, get_args, get_origin

from typing_extensions import ParamSpec, TypeGuard

H = TypeVar("H", bound=Hashable)
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
    return t in (int, float, complex, str, bool, bytes, bytearray, type)


def is_builtin_container(t: Any) -> TypeGuard[ContainerBuiltins[Any]]:
    return t in (list, tuple, dict, set, frozenset)


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

    is_primitive = is_builtin_primitive(t)
    is_container = is_builtin_container(t)
    is_singleton = is_builtin_singleton(t)
    return is_primitive or is_container or is_singleton

if sys.version_info >= (3, 14):
    from typing import evaluate_forward_ref as eval_type
elif sys.version_info >= (3, 13):
    def eval_type(
        value: Union[ForwardRef, GenericAlias, str],
        *,
        globals: Union[dict[str, Any], None] = None,
        locals: Union[Mapping[str, Any], None] = None,
        lenient: bool = False,
    ) -> Any:
        if isinstance(value, str):
            value = ForwardRef(value, is_argument=False)

        try:
            return cast(type[Any], ty_eval_type(value, globals, locals, type_params=tuple()))
        except NameError:
            if not lenient:
                raise
            return value
else:
    def eval_type(
        value: Union[ForwardRef, GenericAlias, str],
        *,
        globals: Union[dict[str, Any], None] = None,
        locals: Union[Mapping[str, Any], None] = None,
        lenient: bool = False,
    ) -> Any:
        if isinstance(value, str):
            value = ForwardRef(value, is_argument=False)

        try:
            return cast(type[Any], ty_eval_type(value, globals, locals))
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


    annotation = eval_type(annotation, globals=gvars, locals=gvars, lenient=True)
    return annotation

def flatten_annotated(typ: Annotated[Any, Any]) -> list[Any]:
    "Annotated[Annotated[T, Ann1, Ann2], Ann3] -> [T, Ann1, Ann2, Ann3]"
    flattened_metadata: list[Any] = []
    _, *metadata = get_args(typ)

    for item in metadata:
        if get_origin(item) is Annotated:
            flattened_metadata.extend(flatten_annotated(item))
        else:
            flattened_metadata.append(item)
    return flattened_metadata