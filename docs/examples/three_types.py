## Builtin types

"""
class without logical behavior
might have meaning in its origin class
but now is context-less data
"""


names: list[str] = ["tom", "jack"]

"""
we know very little about "tom", who is it, 
what can we expect from him?
"""


## model class
class User:
    """
    # these class are representation of concrete object in real-world

    """

    name: str
    age: int
    # birth_date: datetime


# 1. pure data without business context
"""
str, int, subclass of str, int, enum

class PositiveInt(int):
    ...
"""
# 2. data with business context
"""
class BankAccount:
    ...
"""

# 3. function-driven behavior class without business context
"""
class Database:
    ...
"""

# 4. function-driven behavior class with business context
"""
class Repository:
    ...
"""

### IO involved

### high abstract glue classes, functional class.


class UserService:
    """
    used to create user, remove user, change user profile, etc.
    """


### adapters, tailor external services to our needs
"""
These class are highly abstract, behavior / functionality heavy,
mostly has no corredpoing object in real-world, 
or it's modeling our app itself.
"""
