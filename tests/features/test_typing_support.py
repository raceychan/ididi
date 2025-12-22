from typing import Literal, Optional, TypedDict, NewType

from typing_extensions import Unpack

from ididi import Graph


class DataBase:
    ...


class Cache:
    ...


class Config:
    ...


class KWARGS(TypedDict):
    db: DataBase
    cache: Cache


class Extra(KWARGS):
    config: Config


class Repo:
    def __init__(self, **kwargs: Unpack[KWARGS]) -> None:
        self.db = kwargs["db"]
        self.cache = kwargs["cache"]


class SubRepo(Repo):
    def __init__(self, **kwargs: Unpack[Extra]):
        super().__init__(db=kwargs["db"], cache=kwargs["cache"])
        self.config = kwargs["config"]


def test_resolve_unpack():
    dg = Graph()
    repo = dg.resolve(Repo)
    assert isinstance(repo.db, DataBase)
    assert isinstance(repo.cache, Cache)
    subrepo = dg.resolve(SubRepo)
    assert isinstance(subrepo.config, Config)


def test_resolve_literal():
    dg = Graph()

    class User:
        def __init__(self, name: Literal["user"] = "user"):
            self.name = name

    u = dg.resolve(User)
    assert u.name == "user"


def test_optional():
    dg = Graph()

    class UserRepo:
        def __init__(self, db: Optional[DataBase]):
            self.db = db

    assert isinstance(dg.resolve(UserRepo).db, DataBase)




def test_solve_with_new_type():
    from typing import NewType


    class User:
        def __init__(self, user_id: int=1):
            self.user_id = user_id

    MyUser = NewType("MyUser", User)
    IUser = NewType("IUser", User) 

    dg = Graph()

    def my_user_factory() -> MyUser:
        return MyUser(User(user_id=42))
    def iuser_factory() -> IUser:
        return IUser(User(user_id=99))

    user = dg.resolve(my_user_factory)
    assert isinstance(user, User) and user.user_id == 42
    user2 = dg.resolve(iuser_factory)
    assert isinstance(user2, User) and user2.user_id == 99

    
def test_solve_agent_new_types():
    class LLMService: ...
    class Agent: 
        def __init__(self, prompt: str, llm: LLMService):
            self.prompt = prompt
            self.llm = llm

    MathAgent = NewType("MathAgent", Agent)
    EnglishAgent = NewType("EnglishAgent", Agent)

    def math_agent_factory(llm: LLMService) -> MathAgent:
        return MathAgent(Agent(prompt="Solve math problems", llm=llm))

    def english_agent_factory(llm: LLMService) -> EnglishAgent:
        return EnglishAgent(Agent(prompt="Assist with English tasks", llm=llm))

    dg = Graph()
    math_agent = dg.resolve(math_agent_factory)
    assert isinstance(math_agent, Agent)
    assert math_agent.prompt == "Solve math problems"
    english_agent = dg.resolve(english_agent_factory)
    assert isinstance(english_agent, Agent)
    assert english_agent.prompt == "Assist with English tasks"

