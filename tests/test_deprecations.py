import inspect

import pytest

from ididi import DependentNode, use, Ignore
from ididi.errors import DeprecatedError


def _provide_int() -> int:
    return 7





def test_annotated_use_style_no_deprecation():
    # Preferred style: Annotated carries the factory information
    def func(x: int =  use(_provide_int)) -> Ignore[int]:  # type: ignore[valid-type]
        return x

    sig = inspect.signature(func)


    with pytest.raises(DeprecatedError):
        node = DependentNode.from_node(func)
