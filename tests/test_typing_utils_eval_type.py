import sys
from typing import Generic, TypeVar, get_args, get_origin

import pytest

from ididi.utils.typing_utils import eval_type


T = TypeVar("T")


class MyClass(Generic[T]):
    pass


def _eval(expr: str, gvars: dict[str, object], *, lenient: bool = False):
    # `ididi.utils.typing_utils.eval_type` is a shim that can map to different
    # stdlib APIs across Python versions; call it in a signature-tolerant way.
    try:
        return eval_type(expr, globals=gvars, locals=gvars, lenient=lenient)
    except TypeError:
        try:
            return eval_type(expr, globalns=gvars, localns=gvars, lenient=lenient)
        except TypeError:
            return eval_type(expr, gvars, gvars, lenient=lenient)


@pytest.mark.debug
def test_eval_type_keeps_generic_args_for_forwardref():
    gvars = globals()

    resolved = _eval("MyClass[int]", gvars)
    assert get_origin(resolved) is MyClass
    assert get_args(resolved) == (int,)

    resolved_typevar = _eval("MyClass[T]", gvars)
    assert get_origin(resolved_typevar) is MyClass
    assert get_args(resolved_typevar) == (T,)


@pytest.mark.debug
def test_eval_type_lenient_keeps_unresolved_forwardref():
    gvars = globals()

    unresolved = _eval("NotDefined[int]", gvars, lenient=True)
    assert unresolved.__forward_arg__ == "NotDefined[int]"

    with pytest.raises(NameError):
        _eval("NotDefined[int]", gvars, lenient=False)


@pytest.mark.debug
def test_eval_type_generic_superclass_subscript_still_errors():
    gvars = globals()

    # This is a typing invariant: Generic[...] only accepts type variables.
    with pytest.raises(TypeError):
        _eval("Generic[int]", gvars)

    if sys.version_info < (3, 11):
        # Sanity: in older versions the error message can differ; just assert it errors.
        with pytest.raises(TypeError):
            _eval("Generic[str]", gvars)
