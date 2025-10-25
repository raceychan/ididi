import inspect
from typing import Annotated

import pytest

from ididi import use
from ididi._node import Dependencies
from ididi.errors import NotSupportedError


def _provide_int() -> int:
    return 7





def test_annotated_use_style_no_deprecation():
    # Preferred style: Annotated carries the factory information
    def func(x: int =  use(_provide_int)) -> int:  # type: ignore[valid-type]
        return x

    sig = inspect.signature(func)

    with pytest.raises(NotSupportedError):
        Dependencies.from_signature(function=func, signature=sig)
