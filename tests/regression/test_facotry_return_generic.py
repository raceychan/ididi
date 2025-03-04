from typing import Generic, TypeVar

import pytest

from ididi import Graph, Ignore
from ididi.errors import ForwardReferenceNotFoundError, UnsolvableReturnTypeError

dg = Graph()
# Define a generic type variable
T = TypeVar("T")


# Generic base class
class Animal(Generic[T]):
    def __init__(self, value: T):
        self.value = value

    def get_value(self) -> T:
        return self.value


# Concrete implementation
class Dog(Animal[str]):
    def __init__(self, name: str):
        super().__init__(name)

    def bark(self) -> str:
        return f"{self.value} says woof!"


def test_factory_generic_return():
    # Create a Dog instance using the factory
    # Factory class with generic return type
    def animal_factory(animal_type: Ignore[str]) -> Animal[str]:
        return Dog(animal_type)

    dog = dg.resolve(animal_factory, animal_type="Buddy")
    assert isinstance(dog, Animal)
    assert isinstance(dog, Dog)
    assert dog.get_value() == "Buddy"
    assert dog.bark() == "Buddy says woof!"


def test_emptry_return_factory():
    with pytest.raises(UnsolvableReturnTypeError):

        @dg.node
        def animal_factory(animal_type: str): ...

    with pytest.raises(ForwardReferenceNotFoundError):

        @dg.node
        def animal_factory(animtal_type: str = "test") -> "Normal": ...


class UserService: ...


@dg.node
def user_service_factory() -> "UserService": ...


def test_forward_factory():
    dg.node(user_service_factory)
    dg.analyze(user_service_factory)


# Factory class with generic return type
def animal_factory(animal_type: str) -> Animal[str]:
    return Dog(animal_type)


def test_direct_resolve_with_override():
    dg = Graph()
    # this should not raise ididi.errors.UnsolvableDependencyError
    # since animal_type is provided, we should not care about resolvability

    dg.analyze(animal_factory)

    dg.resolve(animal_factory, animal_type="cat")
