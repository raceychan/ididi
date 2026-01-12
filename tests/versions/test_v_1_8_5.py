from typing import Generic, TypeVar
import pytest
from ididi.utils.typing_utils import eval_type

T = TypeVar("T")



class MyGenericClass(Generic[T]):
    pass



@pytest.mark.debug
def test_eval_type_with_type_param():
    eval_type(MyGenericClass[int])
