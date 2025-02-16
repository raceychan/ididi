import typing as ty
from dataclasses import dataclass


@dataclass
class Item:
    price: float
    quantity: int

    @property
    def cost(self) -> float:
        return self.price * self.quantity


@dataclass
class Discount:
    price_off: float


@dataclass
class ShopCart:
    items: ty.Sequence[Item]
    discount: Discount

    def total(self):
        p: float = 0
        for item in self.items:
            p += item.cost * (1 - self.discount.price_off)
        return p


@dataclass
class EconomicShopCart(ShopCart):
    ...


@dataclass
class LuxuryShopCart(ShopCart):
    cart_rent: float

    def total(self):
        return super().total() + self.cart_rent


def create_cart(
    items: ty.Sequence[Item], discount: Discount, *, cart_rent: float | None = None
) -> ShopCart:
    print("Hello, World")

    if cart_rent:
        return LuxuryShopCart(items=items, discount=discount, cart_rent=cart_rent)
    return EconomicShopCart(items=items, discount=discount)


def main():
    # Create economic cart
    items = [Item(price=100, quantity=2)]
    discount = Discount(price_off=0.1)

    economic_cart = create_cart(items, discount)
    print(economic_cart.total())

    # Create luxury cart
    luxury_cart = create_cart(items, discount, cart_rent=50)
    print(luxury_cart.total())


if __name__ == "__main__":
    main()
