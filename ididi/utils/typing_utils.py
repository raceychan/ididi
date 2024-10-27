import inspect
import typing as ty
from typing import _eval_type as ty_eval_type  # type: ignore


def is_builtin_type(t: ty.Any) -> bool:
    return t.__module__ == "builtins" and not issubclass(t, ty.Generic)


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
def first_implementation[
    I
](abstract_types: type[I], implementations: list[type],) -> type[I] | None:
    """
    Find the first concrete implementation of param_type in the given dependencies.
    Returns None if no matching implementation is found.
    """
    matched_deps = (
        dep
        for dep in implementations
        if isinstance(dep, type)
        and isinstance(abstract_types, type)
        and issubclass(dep, abstract_types)
    )
    return next(matched_deps, None)

