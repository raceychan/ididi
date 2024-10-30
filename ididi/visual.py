# from .graph import DependencyGraph
# from .node import DependentNode as DependentNode

# try:
#     from graphviz import Digraph
# except ImportError:
#     pass


# class Visualizer:
#     def __init__(
#         self,
#         graph: DependencyGraph,
#         dot: "Digraph | None" = None,
#         graph_attrs: dict[str, str] | None = None,
#     ):
#         self._dg = graph
#         self._dot = dot
#         self._graph_attrs = graph_attrs

#     def make_graph(
#         self,
#         node_attr: dict[str, str] = {"color": "black"},
#         edge_attr: dict[str, str] = {"color": "black"},
#     ):
#         """
#         # TODO: ignore builtin types
#         Convert DependencyGraph to Graphviz visualization

#         Args:
#             graph: Your DependencyGraph instance
#             output_path: Output file path (without extension)
#             format: Output format ('png', 'svg', 'pdf')
#         """
#         dot = Digraph(comment="Dependency Graph", graph_attr=self._graph_attrs)

#         # Add edges
#         for node in self._dg.nodes.values():
#             node_repr = str(node)
#             for dependency in node.dependency_params:
#                 dependency_repr = str(dependency.dependency)
#                 dot.node(node_repr, node_repr, **node_attr)
#                 dot.edge(node_repr, dependency_repr, **edge_attr)

#         return self.__class__(self._dg, dot, self._graph_attrs)

#     def make_node[
#         T
#     ](self, node: type[T], node_attr: dict[str, str], edge_attr: dict[str, str]):
#         dot = Digraph(comment=f"Dependency Graph {node}", graph_attr=self._graph_attrs)

#         dep_node: DependentNode[T] = self._dg.node(node)

#         node_repr = str(dep_node)
#         for dependency in dep_node.dependency_params:
#             dependency_repr = str(dependency.dependency)
#             dot.node(node_repr, node_repr, **node_attr)
#             dot.edge(node_repr, dependency_repr, **edge_attr)
#         return self.__class__(self._dg, dot, self._graph_attrs)

#     def save(self, output_path: str, format: str = "png"):
#         # Render the graph
#         self._dot.render(output_path, format=format, cleanup=True)

#     def show(self):
#         self._dot.view()
