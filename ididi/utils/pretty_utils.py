from ..graph import DependencyGraph


def pretty_print(graph: DependencyGraph) -> str:
    """
    Returns a string representation of the dependency graph in a tree format.
    Example:
    UserService
    ├── UserRepository
    │   ├── Database
    │   │   └── Config
    │   └── Cache
    │       └── Config
    └── AuthService
        └── Database
            └── Config
    """

    def _build_tree(node_type: type, visited: set[type], prefix: str = "") -> str:
        if node_type in visited:
            return f"{prefix}└── {node_type.__name__} (circular ref)\n"

        visited.add(node_type)
        result = [f"{prefix}└── {node_type.__name__}"]
        dependencies = sorted(graph.dependencies[node_type], key=lambda x: x.__name__)

        for i, dep in enumerate(dependencies):
            is_last = i == len(dependencies) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            result.append(_build_tree(dep, visited.copy(), new_prefix))

        repr_str = "\n".join(result)
        return repr_str

    # Find root nodes (nodes with no dependents or dependents not in graph)
    roots = {
        t
        for t in graph.nodes
        if not graph.dependents[t]
        or all(dep not in graph.nodes for dep in graph.dependents[t])
    }

    if not roots:
        return "(empty graph)"

    result: list[str] = []
    for root in sorted(roots, key=lambda x: x.__name__):
        result.append(f"{root.__name__}")
        deps = sorted(graph.dependencies[root], key=lambda x: x.__name__)
        for i, dep in enumerate(deps):
            is_last = i == len(deps) - 1
            prefix = "    " if is_last else "│   "
            result.append(_build_tree(dep, {root}, prefix))

    return "\n".join(result)
