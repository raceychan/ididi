import inspect

import pytest

from ididi import use
from ididi._node import Dependencies
from ididi.errors import DeprecatedError


def _provide_int() -> int:
    return 7





def test_annotated_use_style_no_deprecation():
    # Preferred style: Annotated carries the factory information
    def func(x: int =  use(_provide_int)) -> int:  # type: ignore[valid-type]
        return x

    sig = inspect.signature(func)

    with pytest.raises(DeprecatedError):
        Dependencies.from_signature(function=func, signature=sig)
