
from dataclasses import dataclass
from typing import Annotated

import pytest

from ididi import Graph, Ignore


@dataclass
class Meta:
    name: str
    version: str

    
def meta(name: str, version: str):
    return Meta(name = name, version = version)


def get_user(q: Annotated[str, meta("query_param", "v1")]) -> Ignore[str]:
    return f"User with query {q}"


def test_meta_annotation():
    graph = Graph()
    graph.node(get_user)
    node = graph.analyze(get_user)
    param = node.dependencies[0]
    assert param.annotation == Annotated[str, meta("query_param", "v1")]

    with pytest.raises(AttributeError):
        param.__dict__

    with pytest.raises(AttributeError):
        node.__dict__

def test_class_annotation():
    @dataclass
    class User:
        id: Annotated[int, meta("user_id", "v1")]
        name: str

    @dataclass
    class Post:
        user: User

    graph = Graph()
    graph.node(Post)
    node = graph.analyze(Post)
    field = node.dependencies[0]
    assert field.name == "user"
