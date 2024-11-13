from typing import TypeVar, Generic, Type
from ididi import DependencyGraph

dg = DependencyGraph()
# Define a generic type variable
T = TypeVar('T')

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

# Factory class with generic return type
@dg.node
def animal_factory(animal_type: str) -> Animal[str]:
    return Dog(animal_type)

# Test cases
def test_factory_generic_return():
    # Create a Dog instance using the factory
    dog = animal_factory("Buddy")
    
    assert isinstance(dog, Animal)
    assert isinstance(dog, Dog)
    assert dog.get_value() == "Buddy"
    assert dog.bark() == "Buddy says woof!"