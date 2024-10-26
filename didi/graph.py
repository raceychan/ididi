import inspect
import typing as ty
from collections import defaultdict
from dataclasses import dataclass, field

from didi.errors import CircularDependencyError
from didi.node import DependencyNode
from didi.utils.typing_utils import get_full_typed_signature, is_builtin_type


@dataclass
class DependencyGraph[T]:
    """
    A dependency DAG (Directed Acyclic Graph) that manages dependency nodes and their relationships.
    Unlike a tree, a node can have multiple parents (dependents).
    """

    nodes: dict[type, DependencyNode] = field(default_factory=dict)
    _resolved_instances: dict[type, ty.Any] = field(default_factory=dict)
    _resolution_stack: set[type] = field(default_factory=set)
    _type_mappings: dict[type, set[type]] = field(
        default_factory=lambda: defaultdict(set)
    )
    # Track who depends on whom
    _dependents: defaultdict[type, set[type]] = field(
        default_factory=lambda: defaultdict(set)
    )
    _dependencies: defaultdict[type, set[type]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def _try_override_factory[I](self, factory: ty.Callable[..., I] | type[I]) -> bool:
        """
        Check if factory is for an existing node type and override if found.
        Returns True if override was performed.
        """
        if not callable(factory) or inspect.isclass(factory):
            return False

        sig = get_full_typed_signature(factory)
        if sig.return_annotation is inspect.Signature.empty:
            return False

        dependent_type = sig.return_annotation
        if dependent_type not in self.nodes:
            return False

        # Update existing node with new factory
        node = self.nodes[dependent_type]
        node.factory = factory
        node.signature = sig
        node.dependencies = DependencyNode.from_node(factory).dependencies
        return True

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
        if self._try_override_factory(factory_or_class):
            return factory_or_class

        node = DependencyNode.from_node(factory_or_class)
        self.register_node(node)
        return factory_or_class

    def register_node(self, node: "DependencyNode") -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """
        if node.dependent in self.nodes:
            return  # Skip if already registered

        # Register main type
        self.nodes[node.dependent] = node

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
        if self._has_cycle():
            # Rollback registration
            self.nodes.pop(node.dependent)
            self._type_mappings[node.dependent].remove(node.dependent)
            self._dependencies.pop(node.dependent)
            for dep_type in self._dependents:
                dep_type.discard(node.dependent)
            raise ValueError("Registration would create a circular dependency")

    def _has_cycle(self) -> bool:
        """
        Detect cycles in the dependency graph using DFS.
        """
        visited = set()
        path = set()

        def visit(node_type: type) -> bool:
            if node_type in path:
                return True  # Cycle detected
            if node_type in visited:
                return False

            path.add(node_type)
            visited.add(node_type)

            # Check all dependencies
            for dep_type in self._dependencies[node_type]:
                if visit(dep_type):
                    return True

            path.remove(node_type)
            return False

        return any(visit(node_type) for node_type in self.nodes)

    def get_dependent_types(self, dependency_type: type) -> set[type]:
        """
        Get all types that depend on the given type.
        """
        return self._dependents[dependency_type]

    def get_dependency_types(self, dependent_type: type) -> set[type]:
        """
        Get all types that the given type depends on.
        """
        return self._dependencies[dependent_type]

    def resolve[I](self, dependency_type: type[I], **overrides: ty.Any) -> I:
        """
        Resolve a dependency and its complete dependency graph.
        Supports dependency overrides for testing.
        """

        if dependency_type in self._resolution_stack:
            raise CircularDependencyError(dependency_type, self._resolution_stack)

        # Use override if provided
        if dependency_type in overrides:
            return overrides[dependency_type]

        # Check if already resolved
        if dependency_type in self._resolved_instances:
            return self._resolved_instances[dependency_type]

        # Handle builtin types
        if is_builtin_type(dependency_type):
            # For builtin types, we just return the override if provided or None
            instance = overrides.get(dependency_type.__name__.lower())
            self._resolved_instances[dependency_type] = instance
            return instance

        # Find concrete implementation
        concrete_type = self._resolve_concrete_type(dependency_type)
        if concrete_type not in self.nodes:
            raise ValueError(f"No registration found for {dependency_type}")

        node = self.nodes[concrete_type]

        # Track resolution stack for cycle detection
        self._resolution_stack.add(dependency_type)
        try:
            # Resolve dependencies first
            deps = {}
            for dep_type in self._dependencies[concrete_type]:
                deps[dep_type.__name__.lower()] = self.resolve(dep_type, **overrides)

            # Create instance
            instance = node.build(**deps, **overrides)
            self._resolved_instances[dependency_type] = instance
            return instance
        finally:
            self._resolution_stack.remove(dependency_type)

    def _resolve_concrete_type(self, abstract_type: type) -> type:
        """
        Resolve abstract type to concrete implementation.
        """
        # If the type is already concrete and registered, return it
        if abstract_type in self.nodes:
            return abstract_type

        implementations = self._type_mappings.get(abstract_type, set())
        if not implementations:
            raise ValueError(f"No implementation found for {abstract_type}")
        if len(implementations) > 1:
            raise ValueError(
                f"Multiple implementations found for {abstract_type}: {implementations}"
            )
        return next(iter(implementations))

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

        for node_type in self.nodes:
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
            if instance and hasattr(instance, "close") and callable(instance.close):
                await instance.close()
        self.reset()

    def __repr__(self) -> str:
        """
        Returns a concise representation of the graph showing:
        - Total number of nodes
        - Number of resolved instances
        - Root nodes (nodes with no dependents)
        - Leaf nodes (nodes with no dependencies)
        """
        roots = {t for t in self.nodes if not self._dependents[t]}
        leaves = {t for t in self.nodes if not self._dependencies[t]}

        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self.nodes)}, "
            f"resolved={len(self._resolved_instances)}, "
            f"roots=[{', '.join(t.__name__ for t in sorted(roots, key=lambda x: x.__name__))}], "
            f"leaves=[{', '.join(t.__name__ for t in sorted(leaves, key=lambda x: x.__name__))}])"
        )


dag = DependencyGraph()
