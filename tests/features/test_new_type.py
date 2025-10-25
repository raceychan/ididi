from typing import Annotated, NewType
from uuid import uuid4

from ididi import Graph, use

# Initialize dependency graph
dg = Graph()

# Define NewTypes
UserID = NewType("UserID", str)
UserName = NewType("UserName", str)
UserAge = NewType("UserAge", int)
AddressID = NewType("AddressID", str)
PostalCode = NewType("PostalCode", str)


# Factory functions
def user_id_factory() -> UserID:
    return UserID(str(uuid4()))


def address_id_factory() -> AddressID:
    return AddressID(str(uuid4()))


def default_name_factory() -> UserName:
    return UserName("Default User")


def default_age_factory() -> UserAge:
    return UserAge(25)


def default_postal_factory() -> PostalCode:
    return PostalCode("12345")


class Address:
    def __init__(
        self,
        postal_code: Annotated[PostalCode, use(default_postal_factory)],
        address_id: Annotated[AddressID, use(address_id_factory)],
    ):
        self.postal_code = postal_code
        self.address_id = address_id


class User:
    def __init__(
        self,
        address: Address,
        name: Annotated[UserName, use(default_name_factory)],
        age: Annotated[UserAge, use(default_age_factory)],
        user_id: Annotated[UserID, use(user_id_factory)],
    ):
        self.name = name
        self.age = age
        self.user_id = user_id
        self.address = address


# Test functions
def test_nested_resolve():
    user = dg.resolve(User)
    assert isinstance(user, User)
    assert isinstance(user.address, Address)
    assert isinstance(user.user_id, str)
    assert isinstance(user.address.address_id, str)
    assert user.age == 25
    assert user.name == "Default User"
    assert user.address.postal_code == "12345"


def test_direct_newtypes():
    user_id = dg.resolve(UserID)

    assert isinstance(user_id, str)

    address = dg.resolve(Address)
    assert isinstance(address.postal_code, str)
    assert isinstance(address.address_id, str)


def test_custom_values():
    custom_user = User(
        name=UserName("Custom Name"),
        age=UserAge(30),
        user_id=UserID("custom-id"),
        address=Address(1, 2),
    )
    assert custom_user.name == "Custom Name"
    assert custom_user.age == 30
    assert custom_user.user_id == "custom-id"


def test_new_type_factory():
    node = dg.search("UserID")
    assert node
    assert node.factory_type != "default"
