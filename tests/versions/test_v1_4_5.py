from ididi.config import IGNORE_PARAM_MARK, USE_FACTORY_MARK
from typing import Annotated
from ididi import Graph, NodeConfig


class User:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    @classmethod
    def from_email(cls, email: str) -> "User":
        return User("user", 1, email)


def test_class_method():
    dg = Graph()

    u = dg.resolve(User.from_email, email="e")
    assert u.name == "user"
    assert u.age == 1


class Config:
    def __init__(self, url: Annotated[str, IGNORE_PARAM_MARK]):
        self.url = url


def get_config() -> Config:
    return Config("asdf")

class DB:
    def __init__(self, config: Annotated[Config, USE_FACTORY_MARK, get_config, NodeConfig()]):
        self.config = config

def test_annotated_mark():
    dg = Graph()
    dg.analyze(Config)
    assert not dg.nodes[Config].dependencies
    db = dg.resolve(DB)
    assert db.config.url == "asdf"
