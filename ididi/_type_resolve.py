"""
This module separates the type resolution logic between utils.typing_utils and the core logic, where this module wraps functions in utils.typing_utils with specific domain logic.
"""

import sys
from collections.abc import AsyncGenerator, Generator
from functools import lru_cache
from inspect import Parameter, Signature
from inspect import isasyncgenfunction as isasyncgenfunction
from inspect import isgeneratorfunction as isgeneratorfunction
from types import GenericAlias, MethodType
from typing import (
    Annotated,
    Any,
    AsyncContextManager,
    Awaitable,
    Callable,
    ContextManager,
    ForwardRef,
    Literal,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
)

from typing_extensions import TypeGuard, Unpack

from .config import CacheMax
from .errors import (
    ForwardReferenceNotFoundError,
    GenericDependencyNotSupportedError,
    UnsolvableReturnTypeError,
)
from .interfaces import AsyncClosable, Closable, INode, P, T
from .utils.typing_utils import T, actualize_strforward, eval_type, is_builtin_type

if sys.version_info >= (3, 10):
    from types import UnionType

    UNION_META = (Union, UnionType)
else:
    UNION_META = (Union,)


SyncResource = Union[ContextManager[Any], Closable]
AsyncResource = Union[AsyncContextManager[Any], AsyncClosable]

IDIDI_USE_FACTORY_MARK = "__ididi_use_factory__"
IDIDI_IGNORE_PARAM_MARK = "__ididi_ignore_param__"

FactoryType = Literal["default", "function", "resource", "aresource"]
# carry this information in node so that resolve does not have to do
# iscontextmanager check

ResolveOrder: dict[FactoryType, int] = {
    "default": 1,
    "function": 2,
    "resource": 3,
    "aresource": 3,
}
# when merge graphs we need to make sure a node with default constructor
# does not override a node with resource / function factory

ExtraUnsolvableTypes: set[Any] = {Any, Literal}

try:
    from typing import TypeAliasType  # type: ignore
except ImportError:
    from typing_extensions import TypeAliasType

ExtraUnsolvableTypes.add(TypeAliasType)


class EmptyInitProtocol(Protocol): ...


def get_typed_params(sig: Signature, gvars: dict[str, Any]) -> list[Parameter]:
    params = sig.parameters.values()
    typed_params = [
        Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=actualize_strforward(param.annotation, gvars),
        )
        for param in params
    ]
    return typed_params


def get_factory_sig_from_cls(cls: type[T]) -> Signature:
    """
    Generate a signature from a class via its __init__ method.
    annotate the return type with the class itself.
    """
    gvars = getattr(cls, "__globals__", {})
    sig = Signature.from_callable(cls.__init__)
    params = get_typed_params(sig, gvars)
    return sig.replace(parameters=params, return_annotation=cls)


def get_typed_signature(
    call: Callable[..., T],
    check_return: bool = False,
) -> Signature:
    """
    Get a typed signature from a factory.
    """
    gvars = getattr(call, "__globals__", {})
    raw_sig = Signature.from_callable(call)

    typed_params = get_typed_params(raw_sig, gvars)
    typed_return = actualize_strforward(raw_sig.return_annotation, gvars)

    if isinstance(typed_return, ForwardRef):
        raise ForwardReferenceNotFoundError(typed_return)
    if check_return and is_unsolvable_type(typed_return):
        raise UnsolvableReturnTypeError(call, typed_return)

    return raw_sig.replace(
        parameters=typed_params,
        return_annotation=typed_return,
    )


def resolve_annotation(annotation: Any) -> type:
    origin = get_origin(annotation) or annotation

    if is_new_type(annotation):
        ntype = resolve_new_type(annotation)
        return ntype

    if origin is Annotated:  # we need to perserve __metadata__
        return annotation

    if origin in (Unpack, Literal):
        return cast(type, origin)

    if origin in (AsyncGenerator, Generator):
        dependent, *_ = get_args(annotation)
        return dependent

    if isinstance(annotation, TypeVar):
        raise GenericDependencyNotSupportedError(annotation)

    if origin in UNION_META:
        union_types = get_args(annotation)
        return first_solvable_type(union_types)

    return origin


def resolve_factory(factory: Callable[..., T]) -> type[T]:
    """
    The dependent type from its factory, based on factory signature.

    should handle Annotate, Generator function, etc.
    """
    sig = get_typed_signature(factory, check_return=True)
    dependent: type[T] = resolve_annotation(sig.return_annotation)
    return dependent


def is_unsolvable_type(t: Any) -> bool:
    """
    Types that are not resolved at type resolving.
    Builtin types are unresolvable, but we might have other cases.

    Every unsolved type is a leaf node in the dependency graph.
    they don't have any dependencies, and can't be built with direct call to factory.

    Examples:
    - builtin types
    """

    return is_builtin_type(t) or t is Signature.empty or t in ExtraUnsolvableTypes


def is_actxmgr_cls(
    t: type[T],
) -> TypeGuard[type[AsyncContextManager[T]]]:
    return issubclass(t, AsyncContextManager)


def is_ctxmgr_cls(
    t: type[T],
) -> TypeGuard[type[ContextManager[T]]]:
    return issubclass(t, ContextManager)


def is_class_or_method(obj: Any) -> bool:
    return isinstance(obj, (type, MethodType, classmethod))


def is_class(
    obj: Union[type[T], Callable[..., Union[T, Awaitable[T]]]]
) -> TypeGuard[type[T]]:
    """
    check if obj is a class, since inspect only checks if obj is a class type.
    """
    origin = get_origin(obj) or obj
    is_type = isinstance(origin, type)
    is_generic_alias = isinstance(obj, GenericAlias)
    return is_type or is_generic_alias


def is_class_with_empty_init(cls: type) -> bool:
    """
    Check if a class has an empty __init__ method.
    e.g.:

    class Person: ...
    class IPerson(typing.Protocol): ...
    """
    is_undefined_init = cls.__init__ is object.__init__
    is_protocol = cls.__init__ is EmptyInitProtocol.__init__
    return cls is type or is_undefined_init or is_protocol


def resolve_forwardref(
    dependent: Union[type[T], Callable[..., T]], ref: ForwardRef
) -> type[T]:
    globalvs = dependent.__init__.__globals__

    try:
        return eval_type(ref, globalvs, globalvs)
    except NameError as e:
        raise ForwardReferenceNotFoundError(ref) from e


def first_solvable_type(types: tuple[Any, ...]) -> type:
    for t in types:
        if not is_unsolvable_type(t):
            return t
    return types[0]


"""
check TypeAlias:

type C = str

type(C) is TypeAliasType

"""


@lru_cache(CacheMax)
def resolve_new_type(annotation: Any) -> type:
    name = getattr(annotation, "__name__")
    tyep_repr = getattr(annotation.__supertype__, "__name__")
    ntype = type(
        f"NewType({name!r}: {tyep_repr})",
        (object,),
        {"super_type": annotation.__supertype__},
    )
    ntype.__name__ = name
    return ntype


def is_new_type(annotation: Any) -> bool:
    return hasattr(annotation, "__supertype__")


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


@lru_cache(CacheMax)
def get_bases(dependent: type) -> tuple[type, ...]:
    if issubclass(dependent, Protocol):
        # -3 excludes Protocol, Gener, object
        bases = cast(type, dependent).__mro__[1:-3]
    else:
        bases = dependent.__mro__[1:-1]
    return bases


def resolve_node_type(dependent: INode[P, T]) -> type[T]:
    if is_class(dependent) or is_new_type(dependent):
        dependent_type = resolve_annotation(dependent)
    else:
        dependent_type = resolve_factory(dependent)

    return cast(type[T], dependent_type)
