import inspect
import typing as ty
from typing import _eval_type as ty_eval_type  # type: ignore


# TODO: write a typeguard to cast builtin types
def is_builtin_type(t: ty.Any) -> bool:
    builtins_types = {
        object,
        int,
        float,
        str,
        bool,
        complex,
        list,
        dict,
        set,
        frozenset,
        tuple,
        bytes,
        bytearray,
        type(None),
    }
    if t is None:
        return True
    elif not isinstance(t, type):
        is_builtin = type(t).__module__ == "builtins"
    else:
        is_builtin = t in builtins_types and not issubclass(t, ty.Generic)

    return is_builtin


def eval_type(
    value: ty.Any,
    globalns: dict[str, ty.Any] | None = None,
    localns: ty.Mapping[str, ty.Any] | None = None,
    *,
    lenient: bool = False,
) -> ty.Any:
    """Evaluate the annotation using the provided namespaces.

    Args:
        value: The value to evaluate. If `None`, it will be replaced by `type[None]`. If an instance
            of `str`, it will be converted to a `ForwardRef`.
        localns: The global namespace to use during annotation evaluation.
        globalns: The local namespace to use during annotation evaluation.
        lenient: Whether to keep unresolvable annotations as is or re-raise the `NameError` exception. Default: re-raise.
    """
    if value is None:
        value = type(None)
    elif isinstance(value, str):
        value = ty.ForwardRef(value, is_argument=False, is_class=True)

    try:
        return ty.cast(type[ty.Any], ty_eval_type(value, globalns, localns))
    except NameError:
        if not lenient:
            raise
        # the point of this function is to be tolerant to this case
        return value


def get_typed_annotation(annotation: ty.Any, globalns: dict[str, ty.Any]) -> ty.Any:
    if isinstance(annotation, str):
        annotation = ty.ForwardRef(annotation)
        annotation = eval_type(annotation, globalns, globalns, lenient=True)
    return annotation


def get_full_typed_signature[T](call: ty.Callable[..., T]) -> inspect.Signature:
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]

    return_annotation = get_typed_annotation(signature.return_annotation, globalns)
    typed_signature = inspect.Signature(
        parameters=typed_params, return_annotation=return_annotation
    )
    return typed_signature


def first_implementation(
    abstract_type: type, implementations: list[type]
) -> type | None:
    """
    Find the first concrete implementation of param_type in the given dependencies.
    Returns None if no matching implementation is found.
    """
    if issubclass(abstract_type, ty.Protocol):
        if not abstract_type._is_runtime_protocol:  # type: ignore
            abstract_type._is_runtime_protocol = True  # type: ignore

    matched_deps = (
        dep
        for dep in implementations
        if isinstance(dep, type)
        and isinstance(abstract_type, type)
        and issubclass(dep, abstract_type)
    )
    return next(matched_deps, None)
