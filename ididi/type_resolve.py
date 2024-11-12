"""
This module separates the type resolution logic between utils.typing_utils and the core logic, where this module wraps functions in utils.typing_utils with specific domain logic.
"""

import collections.abc
import contextlib
import inspect
import types
import typing as ty

from .errors import MissingReturnTypeError
from .utils.typing_utils import get_full_typed_signature, is_builtin_type


class EmptyInitProtocol(ty.Protocol): ...


@ty.runtime_checkable
class Closable(ty.Protocol):
    def close(self) -> None: ...


@ty.runtime_checkable
class AsyncClosable(ty.Protocol):
    async def close(self) -> ty.Coroutine[ty.Any, ty.Any, None]: ...


type Resource = ty.AsyncContextManager[ty.Any] | AsyncClosable


def is_closable(type_: object) -> ty.TypeGuard[AsyncClosable]:
    return isinstance(type_, AsyncClosable)


def get_typed_signature[
    T
](call: ty.Callable[..., T], check_return: bool = False,) -> inspect.Signature:
    """
    Get a typed signature from a factory.
    """
    sig = get_full_typed_signature(call)
    sig_return = sig.return_annotation
    if check_return and sig_return is inspect.Signature.empty:
        raise MissingReturnTypeError(call)
    return sig


def get_sig_origin_return[T](sig_return: ty.Any) -> ty.Any:
    if ty.get_origin(sig_return) in (
        collections.abc.AsyncGenerator,
        collections.abc.Generator,
    ):
        dependent, *_ = ty.get_args(sig_return)
        return dependent
    return sig_return


def is_unresolved_type(t: ty.Any) -> bool:
    """
    Types that are not resolved at type resolving.
    Builtin types are unresolvable, but we might have other cases.

    Every unsolved type is a leaf node in the dependency graph.
    they don't have any dependencies, and can't be built with direct call to factory.

    Examples:
    - builtin types
    """
    return is_builtin_type(t)


# def is_generator[T](t: T) -> ty.TypeGuard[ty.Generator[T, None, None]]:
#     return ty.get_origin(t) is collections.abc.Generator


# def is_async_generator[T](t: T) -> ty.TypeGuard[ty.AsyncGenerator[T, None]]:
#     return ty.get_origin(t) is collections.abc.AsyncGenerator


def is_context_manager[T](t: T) -> ty.TypeGuard[ty.ContextManager[T]]:
    return isinstance(t, contextlib.AbstractContextManager)


def is_async_context_manager[T](t: T) -> ty.TypeGuard[ty.AsyncContextManager[T]]:
    return isinstance(t, contextlib.AbstractAsyncContextManager)


def is_class_or_method(obj: ty.Any) -> bool:
    return isinstance(obj, (type, types.MethodType, classmethod))


def is_class[
    T
](obj: type[T] | ty.Callable[..., T | ty.Awaitable[T]]) -> ty.TypeGuard[type[T]]:
    """
    check if obj is a class, since inspect only checks if obj is a class type.
    """
    origin = ty.get_origin(obj) or obj
    is_type = isinstance(origin, type)
    is_generic_alias = isinstance(obj, types.GenericAlias)
    return is_type or is_generic_alias


def is_function[
    T, **P
](obj: type[T] | ty.Callable[P, T]) -> ty.TypeGuard[ty.Callable[P, T]]:
    return isinstance(obj, types.FunctionType)


def is_class_with_empty_init(cls: type) -> bool:
    """
    Check if a class has an empty __init__ method.
    """
    is_undefined_init = cls.__init__ is object.__init__
    is_protocol = cls.__init__ is EmptyInitProtocol.__init__
    return cls is type or is_undefined_init or is_protocol


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
