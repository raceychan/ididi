from inspect import Parameter, Signature
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ForwardRef,
    Generator,
    Mapping,
    TypeVar,
    Union,
)
from typing import _eval_type as ty_eval_type  # type: ignore
from typing import cast

from typing_extensions import ParamSpec, TypeGuard

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")

Resource = Generator[T, None, None]
AsyncResource = AsyncGenerator[T, None]

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
    return t is None


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


def get_typed_annotation(annotation: Any, globalns: dict[str, Any]) -> Any:
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation, is_argument=False)
        annotation = eval_type(annotation, globalns, globalns, lenient=True)
    return annotation


def get_typed_params(call: Callable[..., T]) -> list[Parameter]:
    signature = Signature.from_callable(call)
    globalns = getattr(call, "__globals__", {})
    globalns.pop("copyright", None)  #
    typed_params = [
        Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]
    return typed_params


def get_full_typed_signature(call: Callable[..., T]) -> Signature:
    """
    Get a full typed signature from a callable.
    check_return: bool

    if check_return is True, raise MissingReturnTypeError if the return type is Signature.emp
    """
    signature = Signature.from_callable(call)

    globalns = getattr(call, "__globals__", {})
    return Signature(
        parameters=get_typed_params(call),
        return_annotation=get_typed_annotation(signature.return_annotation, globalns),
    )


def get_factory_sig_from_cls(cls: type[T]) -> Signature:
    """
    Generate a signature from a class via its __init__ method.
    annotate the return type with the class itself.
    """
    params = get_typed_params(cls.__init__)
    return Signature(parameters=params, return_annotation=cls)
