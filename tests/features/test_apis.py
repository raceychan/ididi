import ididi


class Config:
    def __init__(self, env: str = "prod"):
        self.env = env


class Database:
    def __init__(self, config: Config):
        self.config = config


class UserRepository:
    def __init__(self, db: Database):
        self.db = db


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo


def test_ididi_solve():
    assert isinstance(ididi.solve(UserService), UserService)


@ididi.entry
def main(user_service: UserService):
    assert isinstance(user_service, UserService)
    return user_service


assert isinstance(main(), UserService)
