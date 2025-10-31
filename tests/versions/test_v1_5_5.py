from dataclasses import dataclass


from ididi import Ignore


@dataclass
class User:
    name: str
    age: int


side_effect: list[int] = []


async def get_user() -> Ignore[User]:
    global side_effect
    side_effect.append(1)
    return User("test", 1)







