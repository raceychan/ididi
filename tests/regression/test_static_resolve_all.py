from dataclasses import dataclass

import pytest

from ididi import Graph
from ididi.errors import TopLevelBulitinTypeError, UnsolvableReturnTypeError


@dataclass
class UserCommand:
    user_id: str


@dataclass
class CreateUser(UserCommand):
    user_name: str


@dataclass
class RemoveUser(UserCommand):
    user_name: str


@dataclass
class UpdateUser(UserCommand):
    old_name: str
    new_name: str


def update_user(cmd: UpdateUser) -> str:
    return cmd.new_name


class UserService:
    def __init__(self):
        self.name = "name"

    def create_user(self, cmd: CreateUser) -> str:
        return "hello"

    def remove_user(self, cmd: RemoveUser) -> str:
        return "goodbye"


def test_analyze_nodes():
    dg = Graph()
    dg.node(UserService)
    with pytest.raises(UnsolvableReturnTypeError):
        dg.node(update_user)

    dg.analyze_nodes()


def test_dg_node_builtin():
    dg = Graph()

    with pytest.raises(TopLevelBulitinTypeError):
        dg.node(int)
