from .graph import DependencyGraph
from .node import DependentNode as DependentNode

try:
    from graphviz import Digraph
except ImportError:
    pass


class Visualizer:
    def __init__(
        self,
        graph: DependencyGraph,
        dot: "Digraph | None" = None,
        graph_attrs: dict[str, str] | None = None,
    ):
        self._dg = graph
        self._dot = dot
        self._graph_attrs = graph_attrs

    @property
    def dot(self) -> "Digraph | None":
        return self._dot

    @property
    def view(self) -> "Digraph | None":
        return self.make_graph().dot

    def make_graph(
        self,
        node_attr: dict[str, str] = {"color": "black"},
        edge_attr: dict[str, str] = {"color": "black"},
    ) -> "Visualizer":
        """Converting DependencyGraph to Graphviz visualization

        Args:
            node_attr (dict[str, str], optional): Node attributes. Defaults to {"color": "black"}.
            edge_attr (dict[str, str], optional): Edge attributes. Defaults to {"color": "black"}.

        Returns:
            Visualizer: Visualizer instance
        """
        dot = self._dot or Digraph(
            comment="Dependency Graph", graph_attr=self._graph_attrs
        )

        # Add edges
        for node in self._dg.nodes.values():
            node_repr = str(node.dependent_type.__name__)
            for dependency in node.dependency_params:
                dependency_repr = str(dependency.dependency.dependent_type.__name__)
                dot.node(node_repr, node_repr, **node_attr)
                dot.edge(node_repr, dependency_repr, **edge_attr)

        return self.__class__(self._dg, dot, self._graph_attrs)

    def make_node[
        T
    ](
        self, node: type[T], node_attr: dict[str, str], edge_attr: dict[str, str]
    ) -> "Visualizer":
        """
        Create a graphviz graph for a single node
        """
        dot = self._dot or Digraph(
            comment=f"Dependency Graph {node}", graph_attr=self._graph_attrs
        )
        self._dg.node(node)
        dep_node: DependentNode[T] = self._dg.nodes[node]

        node_repr = str(dep_node.dependent_type.__name__)
        for dependency in dep_node.dependency_params:
            dependency_repr = str(dependency.dependency.dependent_type.__name__)
            dot.node(node_repr, node_repr, **node_attr)
            dot.edge(node_repr, dependency_repr, **edge_attr)

        return self.__class__(self._dg, dot, self._graph_attrs)

    def save(self, output_path: str, format: str = "png") -> None:
        # Render the graph
        self._dot.render(output_path, format=format, cleanup=True)
