import typing as ty

from ididi import DependencyGraph

# import pytest


dg = DependencyGraph()


@dg.node
class ServiceConsumer:
    def __init__(self, service: "ConcreteService"):
        self.service = service


class BaseService(ty.Protocol):
    """Base protocol for services"""

    pass


class ExtendedService(BaseService, ty.Protocol):
    """Extended service protocol"""

    pass


class ConcreteService(ExtendedService):
    """Concrete implementation that will be forward referenced"""

    def __init__(self, name: str = "default"):
        self.name = name


def test_forward_ref_protocol_inheritance():
    """
    Test that forward references with protocol inheritance don't work
    because ForwardDependent has a hardcoded __mro__
    """

    # Even though ConcreteService implements ExtendedService and BaseService,
    # when it's forward referenced, its ForwardDependent.__mro__ is hardcoded to
    # (ForwardDependent, object), so the protocol inheritance is lost

    # This should work if inheritance was properly handled
    dg.resolve(ServiceConsumer)

    @dg.node
    def concrete_service_factory() -> ExtendedService:
        return ConcreteService()

    # Still fails because ForwardDependent doesn't know about the inheritance
    dg.resolve(ServiceConsumer)
