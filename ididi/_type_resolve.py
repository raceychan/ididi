"""
This module separates the type resolution logic between utils.typing_utils and the core logic, where this module wraps functions in utils.typing_utils with specific domain logic.
"""

import sys
from collections.abc import AsyncGenerator, Generator
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from inspect import Signature, isasyncgenfunction, isgeneratorfunction
from types import FunctionType, GenericAlias, MethodType
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

from typing_extensions import ParamSpec, TypeGuard

from ._itypes import AsyncClosable, Closable
from .errors import (
    ForwardReferenceNotFoundError,
    GenericDependencyNotSupportedError,
    UnsolvableReturnTypeError,
)
from .utils.typing_utils import T, eval_type, get_full_typed_signature, is_builtin_type

if sys.version_info >= (3, 10):
    from types import UnionType

    UNION_META = (Union, UnionType)
else:
    UNION_META = (Union,)


SyncResource = Union[ContextManager[Any], Closable]
AsyncResource = Union[AsyncContextManager[Any], AsyncClosable]

IDIDI_INJECT_RESOLVE_MARK = "__ididi_node_mark__"

FactoryType = Literal["default", "function", "resource"]
# carry this information in node so that resolve does not have to do
# iscontextmanager check

ResolveOrder: dict[FactoryType, int] = {
    "default": 1,
    "function": 2,
    "resource": 3,
}
# when merge graphs we need to make sure a node with default constructor
# does not override a node with resource


P = ParamSpec("P")


class EmptyInitProtocol(Protocol): ...


def get_typed_signature(
    call: Callable[..., T],
    check_return: bool = False,
) -> Signature:
    """
    Get a typed signature from a factory.
    """
    sig = get_full_typed_signature(call)
    sig_return: Union[type[T], ForwardRef] = sig.return_annotation
    if isinstance(sig_return, ForwardRef):
        raise ForwardReferenceNotFoundError(sig_return)
    if check_return and is_unresolved_type(sig_return):
        raise UnsolvableReturnTypeError(call, sig_return)
    return sig


def resolve_factory_return(sig_return: type[T]) -> type[T]:
    """
    Get dependent type from a factory return
    """
    if get_origin(sig_return) in (AsyncGenerator, Generator):
        dependent, *_ = get_args(sig_return)
        return dependent
    return sig_return


def resolve_factory(factory: Callable[..., T]) -> type[T]:
    """
    The dependent type from its factory, based on factory signature.

    should handle Annotate, Generator function, etc.
    """
    sig = get_typed_signature(factory, check_return=True)
    dependent: type[T] = resolve_factory_return(sig.return_annotation)
    return dependent


def is_unresolved_type(t: Any) -> bool:
    """
    Types that are not resolved at type resolving.
    Builtin types are unresolvable, but we might have other cases.

    Every unsolved type is a leaf node in the dependency graph.
    they don't have any dependencies, and can't be built with direct call to factory.

    Examples:
    - builtin types
    """
    return is_builtin_type(t) or t is Signature.empty


def is_context_manager(t: T) -> TypeGuard[ContextManager[T]]:
    return isinstance(t, AbstractContextManager)


def is_async_context_manager(t: T) -> TypeGuard[AsyncContextManager[T]]:
    return isinstance(t, AbstractAsyncContextManager)


def is_any_context_manager_clss(
    t: type[T],
) -> TypeGuard[Union[type[ContextManager[T]], type[AsyncContextManager[T]]]]:
    return issubclass(t, (ContextManager, AsyncContextManager))


def is_any_generator(func: Union[Callable[..., T], None]):
    return isgeneratorfunction(func) or isasyncgenfunction(func)


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


def is_function(obj: Union[type[T], Callable[P, T]]) -> TypeGuard[Callable[P, T]]:
    return isinstance(obj, FunctionType)


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
    # if is_function(dependent):
    #     globalvs = dependent.__globals__
    # else:
    globalvs = dependent.__init__.__globals__

    try:
        return eval_type(ref, globalvs, globalvs)
    except NameError as e:
        raise ForwardReferenceNotFoundError(ref) from e


def resolve_annotation(annotation: Any) -> type:
    origin = get_origin(annotation) or annotation
    if origin is Annotated:  # perserve __metadata__
        return annotation

    if isinstance(annotation, TypeVar):
        raise GenericDependencyNotSupportedError(annotation)

    # === Solvable dependency, NOTE: order matters!===

    if origin in UNION_META:
        union_types = get_args(annotation)
        # we don't care which type is correct, it would be provided by the factory
        return union_types[0]
    return origin


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


def get_bases(dependent: type) -> tuple[type, ...]:
    if issubclass(dependent, Protocol):
        # -3 excludes Protocol, Gener, object
        bases = cast(type, dependent).__mro__[1:-3]
    else:
        bases = dependent.__mro__[1:-1]
    return bases
