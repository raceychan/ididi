import typing as ty
from typing import Annotated, TypeVar

from typing_extensions import ParamSpec

R = TypeVar("R")
C = TypeVar("C")
P = ParamSpec("P")

DI = Annotated[R, C]
"""
Allow user to denote the factory of their dependency inside function signature
"""


def user_repo() -> str: ...


# def test(user_repo: DI[str, user_repo]): ...


"""
DI
from typing import Literal

def create_user(user_repo: DI[UserRepo]):
    ...

def uesr_repo() -> str:
    ...

async def main(repo: Literal[uesr_repo]):
    ...
"""
from typing_extensions import Unpack

from ididi import INodeConfig, NodeConfig
from ididi.node import DependentNode


class InlineNode:
    def __init__(self, node: DependentNode[R], default: R):
        self.node = node
        self.default = default

    def __getattribute__(self, name: str) -> ty.Any:
        if not self.default:
            raise Exception("needs to be overriden")


def inject(
    dep: ty.Union[type[R], ty.Callable[..., R]],
    /,
    default: ty.Optional[R] = None,
    **config: Unpack[INodeConfig],
) -> R:
    node = DependentNode.from_node(dep, NodeConfig(**config))
    inline_node = InlineNode(node, default)

    return ty.cast(R, inline_node)


# def di(dep: ty.Callable[P, R]) -> R: ...


class UserRepo: ...


def repo_factory() -> UserRepo: ...


async def update_user(repo: UserRepo = inject(repo_factory, reuse=False)):
    return repo
