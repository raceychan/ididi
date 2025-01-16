import typing as ty
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Item(ABC):
    price: float
    quantity: int

    @property
    @abstractmethod
    def cost(self) -> float: ...


@dataclass
class ItemBase(Item):
    price: float
    quantity: int

    @property
    def cost(self) -> float:
        return self.price * self.quantity


class Discount(ABC):
    price_off: float


@dataclass
class DiscountBase(Discount):
    price_off: float


class AbstractShopCart(ty.Protocol):
    items: ty.Sequence[Item]
    discount: Discount


@dataclass
class ShopCartBase:
    items: ty.Sequence[Item]
    discount: Discount

    def total(self):
        p: float = 0
        for item in self.items:
            p += item.cost * (1 - self.discount.price_off)
        return p


@dataclass
class EconomicShopCartImpl(ShopCartBase): ...


@dataclass
class LuxuryShopCartImpl(ShopCartBase):
    cart_rent: float

    def total(self):
        return super().total() + self.cart_rent


class ShopCartCreator(ABC):
    @abstractmethod
    def create(self, items: ty.Sequence[Item], discount: Discount) -> ShopCartBase: ...


class LuxuryShopCartCreator(ShopCartCreator):
    def __init__(self, cart_rent: float):
        self.cart_rent: float = cart_rent

    def create(
        self, items: ty.Sequence[Item], discount: Discount
    ) -> LuxuryShopCartImpl:
        if discount.price_off > 0.5:
            raise ValueError(
                f"Discount with price off higher than 50% can't be applied to luxury shop cart"
            )
        return LuxuryShopCartImpl(
            items=items, discount=discount, cart_rent=self.cart_rent
        )


class EconomicsShopCartCreator(ShopCartCreator):
    def create(
        self, items: ty.Sequence[Item], discount: Discount
    ) -> EconomicShopCartImpl:
        return EconomicShopCartImpl(items, discount=discount)


class ShopCartFactoryManager:
    def __init__(self):
        self._creator_mapping: dict[str, ShopCartCreator] = {}

    def register_creator(self, cart_type: str, creator: ShopCartCreator):
        self._creator_mapping[cart_type] = creator

    def create_cart(
        self, cart_type: str, items: ty.Sequence[Item], discount: Discount
    ) -> ShopCartBase:
        try:
            return self._creator_mapping[cart_type].create(items, discount)
        except KeyError:
            raise ValueError(f"{cart_type} is not registered with a creator")


class ShopCartFactoryManagerBuilder:
    def __init__(self):
        self._manager = ShopCartFactoryManager()

    def with_economic_cart(self):
        self._manager.register_creator("economic", EconomicsShopCartCreator())
        return self

    def with_luxury_cart(self, cart_rent: float):
        creator = LuxuryShopCartCreator(cart_rent=cart_rent)
        self._manager.register_creator("luxury", creator)
        return self

    def build(self) -> ShopCartFactoryManager:
        return self._manager


class AbstractShoprtCartFactoryManagerBuilderFactory(ABC):
    @abstractmethod
    def _pre_create(self): ...

    @abstractmethod
    def create_builder(self) -> ShopCartFactoryManagerBuilder:
        pass


class ShoprtCartFactoryManagerBuilderFactoryBase(
    AbstractShoprtCartFactoryManagerBuilderFactory
):
    """Base implementation of AbstractShoprtCartFactoryManagerBuilderFactory."""

    @abstractmethod
    def _pre_create(self): ...

    def create_builder(self) -> ShopCartFactoryManagerBuilder:
        self._pre_create()
        return ShopCartFactoryManagerBuilder()


class ShoprtCartFactoryManagerBuilderFactoryimpl(
    ShoprtCartFactoryManagerBuilderFactoryBase
):
    """Concrete implementation of the factory."""

    def _pre_create(self):
        print("Hello, World!")


class Application:
    def run(self):
        # Build the factory manager
        builder_factory = ShoprtCartFactoryManagerBuilderFactoryimpl()
        builder = builder_factory.create_builder()
        factory_manager = (
            builder.with_economic_cart().with_luxury_cart(cart_rent=50.0).build()
        )

        # Create economic cart
        items = [ItemBase(price=100, quantity=2)]
        discount = DiscountBase(price_off=0.1)
        economic_cart = factory_manager.create_cart("economic", items, discount)
        print(economic_cart.total())

        # Create luxury cart
        luxury_cart = factory_manager.create_cart("luxury", items, discount)
        print(luxury_cart.total())


if __name__ == "__main__":
    Application().run()
