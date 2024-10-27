import inspect
import typing as ty
from collections import defaultdict

from ididi.errors import (
    CircularDependencyDetectedError,
    MissingImplementationError,
    MultipleImplementationsError,
    TopLevelBulitinTypeError,
    UnregisteredTypeError,
)
from ididi.node import DependencyNode
from ididi.utils.typing_utils import get_full_typed_signature, is_builtin_type

type GraphNodes[I] = dict[type[I], DependencyNode[I]]
type ResolvedInstances[I] = dict[type[I], I]
type TypeMappings[I] = dict[type[I], set[type[I]]]
type NodeRelations[I] = defaultdict[type[I], set[type[I]]]


class DependencyGraph:
    """
    A dependency DAG (Directed Acyclic Graph) that manages dependency nodes and their relationships.
    Unlike a tree, a node can have multiple parents (dependents).
    """

    _nodes: GraphNodes[ty.Any]
    _resolved_instances: ResolvedInstances[ty.Any]
    _resolution_stack: set[type]
    _type_mappings: TypeMappings[ty.Any]
    _dependents: NodeRelations[ty.Any]
    _dependencies: NodeRelations[ty.Any]

    def __init__(self):
        self._nodes = {}
        self._resolved_instances = {}
        self._resolution_stack = set()
        self._type_mappings = defaultdict(set)

        # Track who depends on whom
        self._dependents = defaultdict(set)
        self._dependencies = defaultdict(set)

    def __repr__(self) -> str:
        """
        Returns a concise representation of the graph showing:
        - Total number of nodes
        - Number of resolved instances
        - Root nodes (nodes with no dependents)
        - Leaf nodes (nodes with no dependencies)
        """
        roots = {t for t in self._nodes if not self._dependents[t]}
        leaves = {t for t in self._nodes if not self._dependencies[t]}

        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_instances)}, "
            f"roots=[{', '.join(t.__name__ for t in sorted(roots, key=lambda x: x.__name__))}], "
            f"leaves=[{', '.join(t.__name__ for t in sorted(leaves, key=lambda x: x.__name__))}])"
        )

    def _resolve_concrete_type[I](self, abstract_type: type[I]) -> type[I]:
        """
        Resolve abstract type to concrete implementation.
        """
        # If the type is already concrete and registered, return it
        if abstract_type in self._nodes:
            return abstract_type

        implementations = self._type_mappings.get(abstract_type, set())
        if not implementations:
            raise MissingImplementationError(abstract_type)
        if len(implementations) > 1:
            raise MultipleImplementationsError(abstract_type, implementations)

        return next(iter(implementations))

    def _has_cycle(self) -> bool:
        """
        Detect cycles in the dependency graph using DFS.
        Raises CircularDependencyDetectedError if a cycle is found.
        """
        visited = set()
        path = set()
        path_list: list[type] = []  # Keep list just for cycle reporting

        def visit(node_type: type) -> None:
            if node_type in path:  # O(1)
                # For error reporting only
                cycle_start = path_list.index(node_type)  # Only called when cycle found
                cycle_path = path_list[cycle_start:] + [node_type]
                raise CircularDependencyDetectedError(cycle_path)

            if node_type in visited:
                return

            path.add(node_type)  # O(1)
            path_list.append(node_type)  # O(1)
            visited.add(node_type)  # O(1)

            for dep_type in self._dependencies[node_type]:
                visit(dep_type)

            path.remove(node_type)  # O(1)
            path_list.pop()  # O(1)

        for node_type in self._nodes:
            visit(node_type)

        return False

    def remove_node(self, node: DependencyNode) -> None:
        """
        Remove a node from the graph.
        """
        self._nodes.pop(node.dependent)
        self._type_mappings[node.dependent].remove(node.dependent)
        self._dependencies.pop(node.dependent)
        for dep_type in self._dependents:
            dep_type.discard(node.dependent)

    def register_node(self, node: "DependencyNode") -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """
        if node.dependent in self._nodes:
            return  # Skip if already registered

        # Register main type
        self._nodes[node.dependent] = node

        # Update type mappings for dependency resolution
        self._type_mappings[node.dependent].add(node.dependent)

        for base in inspect.getmro(node.dependent)[1:]:
            if base is not object:
                self._type_mappings[base].add(node.dependent)

        # Recursively register dependencies first
        for dep_node in node.dependencies:
            if not is_builtin_type(dep_node.dependent):
                self.register_node(dep_node)

        # Update dependency relationships
        for dep_node in node.dependencies:
            dep_type = dep_node.dependent
            self._dependencies[node.dependent].add(dep_type)
            self._dependents[dep_type].add(node.dependent)

        # Verify no cycles were introduced
        try:
            self._has_cycle()
        except CircularDependencyDetectedError:
            self.remove_node(node)

    def get_dependent_types(self, dependency_type: type) -> set[type]:
        """
        Get all types that depend on the given type.
        """
        return self._dependents[dependency_type]

    def get_dependency_types[I](self, dependent_type: type[I]) -> set[type[I]]:
        """
        Get all types that the given type depends on.
        """
        return self._dependencies[dependent_type]

    def resolve[I](self, dependency_type: type[I], **overrides: ty.Any) -> I:
        """
        Resolve a dependency and its complete dependency graph.
        Supports dependency overrides for testing.
        """
        if is_builtin_type(dependency_type):
            raise TopLevelBulitinTypeError(dependency_type)

        # if dependency_type in self._resolution_stack:
        #     raise CircularDependencyError(dependency_type, self._resolution_stack)

        # Use override if provided
        if dependency_type.__name__ in overrides:
            return overrides[dependency_type.__name__]
        # Check if already resolved
        if dependency_type in self._resolved_instances:
            return self._resolved_instances[dependency_type]

        # Find concrete implementation
        concrete_type = self._resolve_concrete_type(dependency_type)
        try:
            node = self._nodes[concrete_type]
        except KeyError:
            raise UnregisteredTypeError(concrete_type)

        # Track resolution stack for cycle detection
        self._resolution_stack.add(dependency_type)
        try:
            # Resolve dependencies first
            deps = {}
            for dep_type in self._dependencies[concrete_type]:
                if is_builtin_type(dep_type):
                    continue
                deps[dep_type.__name__] = self.resolve(dep_type, **overrides)
            instance = node.build(**deps, **overrides)
            self._resolved_instances[dependency_type] = instance
            return instance
        finally:
            self._resolution_stack.remove(dependency_type)

    def get_initialization_order(self) -> list[type]:
        """
        Return types in dependency order (topological sort).
        """
        order = []
        visited = set()

        def visit(node_type: type):
            if node_type in visited:
                return

            visited.add(node_type)
            for dep_type in self._dependencies[node_type]:
                visit(dep_type)

            order.append(node_type)

        for node_type in self._nodes:
            visit(node_type)

        return order

    def reset(self) -> None:
        """
        Clear all resolved instances while maintaining registrations.
        """
        self._resolved_instances.clear()
        self._resolution_stack.clear()

    async def aclose(self) -> None:
        """
        Close any resources held by resolved instances.
        Closes in reverse initialization order.
        """
        close_order = reversed(self.get_initialization_order())
        for type_ in close_order:
            instance = self._resolved_instances.get(type_)
            if not instance:
                continue
            if self.is_resource(instance):
                await instance.close()
        self.reset()

    def is_resource(self, type_: type) -> bool:
        return hasattr(type_, "close") and callable(type_.close)

    @ty.overload
    def node[I](self, factory_or_class: type[I]) -> type[I]: ...

    @ty.overload
    def node[I](self, factory_or_class: ty.Callable[..., I]) -> ty.Callable[..., I]: ...

    def node[
        I
    ](self, factory_or_class: ty.Callable[..., I] | type[I]) -> (
        ty.Callable[..., I] | type[I]
    ):
        """
        Decorator to register a node in the dependency graph.
        Can be used with both factory functions and classes.
        If decorating a factory function that returns an existing type,
        it will override the factory for that type.
        """
        dependent_type = get_full_typed_signature(factory_or_class).return_annotation
        is_factory = callable(factory_or_class) and not inspect.isclass(
            factory_or_class
        )
        has_return_type = dependent_type is not inspect.Signature.empty
        is_registered_node = dependent_type in self._nodes
        is_factory_override = is_factory and has_return_type and is_registered_node

        if is_factory_override:
            node = DependencyNode.from_node(factory_or_class)
            self._nodes[dependent_type] = node
        else:
            node = DependencyNode.from_node(factory_or_class)
            self.register_node(node)
        return factory_or_class
