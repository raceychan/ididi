from typing import Union

from ididi._type_resolve import is_unsolvable_type


class User:
    id: int
    name: str

def test_is_unsolvable_type():
    assert is_unsolvable_type(str)
    assert is_unsolvable_type(int)
    assert is_unsolvable_type(list[str])
    assert is_unsolvable_type(dict[int, float])
    assert is_unsolvable_type(tuple[int, ...])
    assert is_unsolvable_type(set[bytes])
    assert is_unsolvable_type(Union[str, None])
    assert not is_unsolvable_type(User)