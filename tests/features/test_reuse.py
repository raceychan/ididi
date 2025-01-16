from ididi.graph import DependencyGraph

dg = DependencyGraph()


class Config:
    def __init__(self, value: str = "config"):
        self.value = value


def test_config_eq():
    ng = DependencyGraph()
    ng.node(Config)
    c1 = ng.nodes[Config].config
    ng.remove_dependent(Config)

    ng.node(Config, reuse=False)
    c2 = ng.nodes[Config].config
    assert c1 != 1
    assert c1 != c2


class Engine:
    def __init__(self, config: Config):
        self.config = config


@dg.node(reuse=False)
class Database:
    def __init__(self, engine: Engine):
        self.engine = engine


def test_reuse():
    @dg.node(reuse=False)
    class Service:
        def __init__(self, database: Database):
            self.database = database

    service1 = dg.resolve(Service)
    service2 = dg.resolve(Service)
    assert id(service2) != id(service1)
    assert id(service2.database) != id(service1.database)

    @dg.node(reuse=True)
    def service_factory() -> Service:
        return Service(Database(Engine(Config())))

    assert service_factory

    service3 = dg.resolve(Service)
    service4 = dg.resolve(Service)
    assert id(service3) == id(service4)
    assert id(service2.database) != id(service1.database)
