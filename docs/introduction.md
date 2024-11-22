# Introduction

## Terminology

> what do we mean by these words

`dependent`: a class, or a function that requires arguments to be built/called.

`dependency`: an object that is required by a dependent.

`factory`: a function that is used to build the dependent, in case where dependent is a function, factory is the dependent.

`resource`: a dependent that implements the contextlib.AbstractAsync/ContextManager, or has an async/sync generator as its factory, is considered a resource.

`static resolve`: resursively build node from dependent type, but does not create the instance of the dependent type.

`resolve`: recursively resolve the dependent and its dependencies, then create an instance of the dependent type.

`entry`: a special type of node, where it has no dependents and its factory is itself.

`dg`: an instance of the DependencyGraph class.

## What and why

### What is Dependency Injection ?

If a class requires other classes as its attributes, then these attributes are regarded as dependencies of the class, and the class requiring them is called a dependent.

```python
class Downloader:
    def __init__(self, session: requests.Session):
        self.session = session
```

Here, `Downloader` is a dependent, with `requests.Session` being its dependency.

Dependency injection means dynamically constructing the instances of these dependency classes and then pass them to the dependent class.

the same class without dependency injection looks like this:

```python
class Downloader:
    def __init__(self):
        self.session = requests.Session(url=configured_url, timeout=configured_timeout)
```

Now, since `requests.Session` is automatically built with `Downloader`, it would be difficult to change the behavior of `requests.Session` at runtime.

### Why do we need it?

There are actually a few reasons why you might not need it, the most fundamental one being your code does not need reuseability and flexibility.

1. If you are writing a script that only runs when you menually execute it, and it is often easier to rewrite the whole script than to modify it,
then it probably more efficient to program everything hard-coded. This is actually a common use case of python,
DEVOPS, DataAnalysts, etc.

For example, you can actually modify the dependencies of a class at runtime.

```python
class Downloader:
    ...

downloader = Downloader()
downloader.session = requests.Session(url=configured_url, timeout=configured_timeout)
```

However, this creates a few problems:

- It is error-prone, you might forget to modify the dependencies, or you might modify the dependencies in the wrong order.
- It is not typesafe, you might pass the wrong type of dependencies to the class.
- It is hard to track when the dependencies are modified.

Dependency injection enables you to extend the dependencies of a class without modifying the class itself, which increases the flexibility and reusability of the class.

### Do we need a DI framework?

Not necessarily.

You will be doing just fine using menual dependency injection, as long as the number of dependencies in your app stays within a managable range.

It gets more and more helpful once your your dependency graph starts getting more complicated,
For example, you might have something like this in your app, where you menually inject dependencies into dependent.

Let's see an example:


```python title="factory.py"
from .infra import * 

def auth_service_factory(
    settings: Settings,
) -> AuthService:
    connect_args = (
        settings.db.connect_args.model_dump()
    )
    execution_options = (
        settings.db.execution_options.model_dump()
    )
    engine = engine_factory(
        db_url=settings.db.DB_URL,
        echo=settings.db.ENGINE_ECHO,
        isolation_level=settings.db.ISOLATION_LEVEL,
        pool_pre_ping=True,
        connect_args=connect_args,
        execution_options=execution_options,
    )
    async_engine = sqlalchemy.ext.asyncio.AsyncEngine(engine)
    db = AsyncDatabase(async_engine)
    cache = RedisCache[str].build(
        url=config.URL,
        keyspace=config.keyspaces.APP,
        socket_timeout=config.SOCKET_TIMEOUT,
        decode_responses=config.DECODE_RESPONSES,
        max_connections=config.MAX_CONNECTIONS,
    )
    token_registry = TokenRegistry(cahce=cache, 
        token_bucket=TokenBucket(cache, key=Settings.redis.bucket_key)
    )
    uow = UnitOfWork(db)
    encryptor = Encryptor(
        secret_key=settings.security.SECRET_KEY.get_secret_value(),
        algorithm=settings.security.ALGORITHM,  
    )
    eventstore =  EventStore(uow)
    auth_repo = AuthRepository(db)
    auth_service = AuthService(
        auth_repo=auth_repo,
        token_registry=token_registry,
        encryptor=encryptor,
        eventstore=eventstore,
        security_settings=settings.security,
    )
    return auth_service
```


But then you realize that some of these dependencies should be shared across your services,
for example, auth repo might be needed by both AuthService and UserService, or even more

You might also need to menually create and manage scope as some resources should be accessed/shared only whtin a certain scope, e.g., a request.

### isn't dependency injection a java thing?

<details>
    <summary>
    java like python code
    </summary>

```py
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
```
</details>


the same logic, but pythonic, with only 1/3 LOC.

<details>
    <summary>
    pythonic python code
    </summary>

```py
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
class EconomicShopCart(ShopCart): ...


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
```

</details>
