import inspect
import typing as ty
from typing import _eval_type as ty_eval_type  # type: ignore

import typing_extensions as tyex

T = ty.TypeVar("T")
R = ty.TypeVar("R")
P = tyex.ParamSpec("P")

PrimitiveBuiltins = type[ty.Union[int, float, complex, str, bool, bytes, bytearray]]
ContainerBuiltins = type[
    ty.Union[list[T], tuple[T, ...], dict[ty.Any, T], set[T], frozenset[T]]
]
BuiltinSingleton = type[None]


def is_builtin_primitive(t: ty.Any) -> tyex.TypeGuard[PrimitiveBuiltins]:
    return t in {int, float, complex, str, bool, bytes, bytearray, type}


def is_builtin_container(t: ty.Any) -> tyex.TypeGuard[ContainerBuiltins[ty.Any]]:
    return t in {list, tuple, dict, set, frozenset}


def is_builtin_singleton(t: ty.Any) -> tyex.TypeGuard[BuiltinSingleton]:
    return t is None


def is_builtin_type(
    t: ty.Any,
) -> tyex.TypeGuard[
    ty.Union[PrimitiveBuiltins, ContainerBuiltins[ty.Any], BuiltinSingleton]
]:
    """
    Builtin types are ignored at type resolving.
    It must be provided by default value, or use a factory to override it.
    typing.Any is also ignored at type resolving.
    """

    # TODO: we might name this as 'is_unresolved_type' as corner cases increase.
    is_primitive = is_builtin_primitive(t)
    is_container = is_builtin_container(t)
    is_singleton = is_builtin_singleton(t)
    return is_primitive or is_container or is_singleton or (t is ty.Any)


def eval_type(
    value: ty.ForwardRef,
    globalns: ty.Union[dict[str, ty.Any], None] = None,
    localns: ty.Union[ty.Mapping[str, ty.Any], None] = None,
    *,
    lenient: bool = False,
) -> ty.Any:
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
        return ty.cast(type[ty.Any], ty_eval_type(value, globalns, localns))
    except NameError:
        if not lenient:
            raise
        return value


def get_typed_annotation(annotation: ty.Any, globalns: dict[str, ty.Any]) -> ty.Any:
    if isinstance(annotation, str):
        annotation = ty.ForwardRef(annotation, is_argument=False)
        annotation = eval_type(annotation, globalns, globalns, lenient=True)
    return annotation


def get_typed_params(call: ty.Callable[..., T]) -> list[inspect.Parameter]:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    globalns.pop("copyright", None)  #
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]
    return typed_params


def get_full_typed_signature(call: ty.Callable[..., T]) -> inspect.Signature:
    """
    Get a full typed signature from a callable.
    check_return: bool

    if check_return is True, raise MissingReturnTypeError if the return type is inspect.Signature.empty.
    """
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    return inspect.Signature(
        parameters=get_typed_params(call),
        return_annotation=get_typed_annotation(signature.return_annotation, globalns),
    )


def get_factory_sig_from_cls(cls: type[T]) -> inspect.Signature:
    """
    Generate a signature from a class via its __init__ method.
    annotate the return type with the class itself.
    """
    params = get_typed_params(cls.__init__)
    return inspect.Signature(parameters=params, return_annotation=cls)
