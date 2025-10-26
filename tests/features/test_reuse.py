from typing import Annotated

from ididi import Graph, use

dg = Graph()


class Config:
    def __init__(self, value: str = "config"):
        self.value = value


def test_config_eq():
    ng = Graph()
    ng.node(Config)
    c1 = ng.nodes[Config].reuse
    ng.remove_dependent(Config)

    ng.node(use(Config, reuse=True))
    c2 = ng.nodes[Config].reuse
    assert c1 != 1
    assert c1 != c2


def test_remove_annotated():
    def config_factory() -> Config:
        ...

    dg = Graph()
    dg.node(config_factory)
    dg.remove_dependent(use(config_factory))


class Engine:
    def __init__(self, config: Config):
        self.config = config


class Database:
    def __init__(self, engine: Engine):
        self.engine = engine


def test_reuse():

    class Service:
        def __init__(self, database: Annotated[Database, use(reuse=False)]):
            self.database = database

    service1 = dg.resolve(Service)
    service2 = dg.resolve(Service)
    assert id(service2) != id(service1)
    assert id(service2.database) != id(service1.database)

    def service_factory() -> Annotated[Service, use(reuse=True)]:
        return Service(Database(Engine(Config())))

    assert service_factory

    service3 = dg.resolve(service_factory)
    service4 = dg.resolve(service_factory)
    assert id(service3) == id(service4)
    assert id(service2.database) != id(service1.database)
