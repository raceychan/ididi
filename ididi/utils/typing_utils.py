import inspect
import typing as ty
from typing import _eval_type as ty_eval_type  # type: ignore

type PrimitiveBuiltins = type[int | float | complex | str | bool | bytes | bytearray]
type ContainerBuiltins[T] = type[
    list[T] | tuple[T, ...] | dict[ty.Any, T] | set[T] | frozenset[T]
]
type BuiltinSingleton = type[None]


def is_builtin_primitive(t: ty.Any) -> ty.TypeGuard[PrimitiveBuiltins]:
    return t in {int, float, complex, str, bool, bytes, bytearray}


def is_builtin_container(t: ty.Any) -> ty.TypeGuard[ContainerBuiltins[ty.Any]]:
    return t in {list, tuple, dict, set, frozenset}


def is_builtin_singleton(t: ty.Any) -> ty.TypeGuard[BuiltinSingleton]:
    return t is None


def is_builtin_type(
    t: ty.Any,
) -> ty.TypeGuard[PrimitiveBuiltins | ContainerBuiltins[ty.Any] | BuiltinSingleton]:
    """
    Builtin types are ignored at type resolving.
    It must be provided by default value, or use a factory to override it.
    typing.Any is also ignored at type resolving.
    """

    # TODO: we might name this as unresolved_type as corner cases increase.
    is_primitive = is_builtin_primitive(t)
    is_container = is_builtin_container(t)
    is_singleton = is_builtin_singleton(t)
    return is_primitive or is_container or is_singleton or (t is ty.Any)


def is_unresolved_type(t: ty.Any) -> ty.TypeGuard[ty.Any]:
    """
    Types that are not resolved at type resolving.
    Builtin types are unresolvable, but we might have other cases.

    Every unsolved type is a leaf node in the dependency graph.
    they don't have any dependencies, and can't be built with direct call to factory.

    Examples:
    - builtin types
    - `typing.Any`
    - `typing.Union[typing.Any, int]`
    - `typing.Optional[typing.Any]`
    - `typing.Callable[[], int]`
    """
    raise NotImplementedError


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


@ty.runtime_checkable
class Closable(ty.Protocol):
    def close(self) -> None: ...


@ty.runtime_checkable
class AsyncClosable(ty.Protocol):
    async def close(self) -> ty.Coroutine[ty.Any, ty.Any, None]: ...


type Resource = ty.AsyncContextManager[ty.Any] | AsyncClosable


def is_closable(type_: object) -> ty.TypeGuard[AsyncClosable]:
    return isinstance(type_, AsyncClosable)


def is_async_context_manager(
    type_: object,
) -> ty.TypeGuard[ty.AsyncContextManager[ty.Any]]:
    return isinstance(type_, ty.AsyncContextManager)


# def is_resource(instance: ty.Any) -> ty.TypeGuard[Resource]:
#     return is_closable(instance)
