from ididi.graph import DependencyGraph
from ididi.visual import Visualizer

if __name__ == "__main__":
    dag = DependencyGraph()

    class Config:
        def __init__(self, env: str = "prod"):
            self.env = env

    class Database:
        def __init__(self, config: Config):
            self.config = config

    class Cache:
        def __init__(self, config: Config):
            self.config = config

    class UserRepository:
        def __init__(self, db: Database, cache: Cache):
            self.db = db
            self.cache = cache

    class AuthService:
        def __init__(self, db: Database):
            self.db = db

    @dag.node
    class UserService:
        def __init__(self, repo: UserRepository, auth: AuthService, name: str = "user"):
            self.repo = repo
            self.auth = auth

    Visualizer(dag).visualize_dependency_graph()
