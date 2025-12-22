from typing_extensions import TypeAliasType

from ididi._type_resolve import resolve_annotation


def test_resolve_annotation_handles_type_alias_type():
    Alias = TypeAliasType("Alias", str)
    assert resolve_annotation(Alias) is str
