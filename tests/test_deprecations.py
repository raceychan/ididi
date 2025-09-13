import inspect
from typing import Annotated

import pytest

from ididi import use
from ididi._node import Dependencies


def _provide_int() -> int:
    return 7


def test_default_use_style_emits_deprecation_warning():
    # Deprecated style: default argument carries the factory information
    def func(x: int = use(_provide_int)) -> int:
        return x

    sig = inspect.signature(func)

    with pytest.warns(DeprecationWarning):
        deps = Dependencies.from_signature(function=func, signature=sig)
        assert "x" in deps


def test_annotated_use_style_no_deprecation():
    # Preferred style: Annotated carries the factory information
    def func(x: Annotated[int, use(_provide_int)]) -> int:  # type: ignore[valid-type]
        return x

    sig = inspect.signature(func)

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        deps = Dependencies.from_signature(function=func, signature=sig)
        assert "x" in deps
        assert not any(msg.category is DeprecationWarning for msg in w)
