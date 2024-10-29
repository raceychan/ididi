from graphviz import Digraph

from .ididi.graph import DependencyGraph


def visualize_dependency_graph(
    graph: DependencyGraph, output_path: str = "dependency_graph", format: str = "png"
):
    """
    # TODO: ignore builtin types
    Convert DependencyGraph to Graphviz visualization

    Args:
        graph: Your DependencyGraph instance
        output_path: Output file path (without extension)
        format: Output format ('png', 'svg', 'pdf')
    """
    dot = Digraph(comment="Dependency Graph")
    dot.attr(rankdir="LR")  # Left to Right layout

    # Add nodes
    for node in graph.nodes:
        node_name = node.__name__ if hasattr(node, "__name__") else str(node)
        dot.node(str(id(node)), node_name)

    # Add edges
    for node in graph.nodes.values():
        for dependency in node.dependency_params:
            dot.edge(str(id(node)), str(id(dependency.dependency.dependent)))

    # Render the graph
    dot.render(output_path, format=format, cleanup=True)


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

    visualize_dependency_graph(dag)
