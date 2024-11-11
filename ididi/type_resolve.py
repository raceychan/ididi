"""
This module separates the type resolution logic between utils.typing_utils and the core logic, where this module wraps functions in utils.typing_utils with specific domain logic.
"""

import collections.abc
import inspect
import typing as ty

from .errors import MissingReturnTypeError
from .utils.typing_utils import get_full_typed_signature, is_builtin_type
from .utils.typing_utils import is_function as is_function


def get_typed_signature[
    T
](call: ty.Callable[..., T], check_return: bool = False) -> inspect.Signature:
    """
    Get a typed signature from a factory.
    """
    sig = get_full_typed_signature(call)
    sig_return = sig.return_annotation
    if check_return and sig_return is inspect.Signature.empty:
        raise MissingReturnTypeError(call)

    if ty.get_origin(sig_return) is collections.abc.AsyncGenerator:
        dependent, *_ = ty.get_args(sig_return)
        sig = sig.replace(return_annotation=dependent)

    return sig


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


def is_async_gen[T](t: T) -> ty.TypeGuard[ty.AsyncGenerator[T, ty.Any]]:
    return isinstance(t, collections.abc.AsyncGenerator)
