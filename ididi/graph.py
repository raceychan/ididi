import inspect
import typing as ty
from collections import defaultdict
from types import MappingProxyType

from .errors import (
    MissingImplementationError,
    MultipleImplementationsError,
    TopLevelBulitinTypeError,
    UnregisteredTypeError,
)
from .node import AbstractDependent, DependentNode, ForwardDependent
from .types import (
    GraphNodes,
    GraphNodesView,
    NodeDependent,
    ResolvedInstances,
    TypeMappings,
    TypeMappingView,
)
from .utils.typing_utils import get_full_typed_signature, is_builtin_type, is_closable


class DependencyGraph:
    """
    ### Description:
    A dependency DAG (Directed Acyclic Graph) that manages dependency nodes and their relationships.

    ### Attributes:
    #### nodes: dict[type, DependentNode]
    mapping a type to its corresponding node

    #### resolved_instances: dict[type, object]
    mapping a type to its resolved instance, to be used for caching

    #### type_mapping: dict[type, list[type]]
    mapping abstract type to its implementations
    """

    # core
    _nodes: GraphNodes[ty.Any]
    _resolved_instances: ResolvedInstances
    _type_mappings: TypeMappings

    def __init__(self):
        self._nodes = {}
        self._resolved_instances = {}
        self._type_mappings = defaultdict(list)

    def __repr__(self) -> str:
        """
        Returns a concise representation of the graph showing:
        - Total number of nodes
        - Number of resolved instances
        - Root nodes (nodes with no dependents)
        - Leaf nodes (nodes with no dependencies)
        """
        roots = {t for t in self._nodes if not self.get_dependent_types(t)}
        leaves = {t for t in self._nodes if not self.get_dependency_types(t)}

        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolved_instances)}, "
            f"roots=[{', '.join(t.__name__ for t in sorted(roots, key=lambda x: x.__name__))}], "
            f"leaves=[{', '.join(t.__name__ for t in sorted(leaves, key=lambda x: x.__name__))}])"
        )

    @property
    def nodes(self) -> GraphNodesView[type]:
        return MappingProxyType(self._nodes)

    @property
    def type_mappings(self) -> TypeMappingView:
        return MappingProxyType(self._type_mappings)

    def _resolve_concrete_type(self, abstract_type: type) -> type:
        """
        Resolve abstract type to concrete implementation.
        """
        # If the type is already concrete and registered, return it
        if abstract_type in self._nodes:
            return abstract_type

        implementations: list[type] = self._type_mappings.get(abstract_type, [])

        if not implementations:
            raise MissingImplementationError(abstract_type)
        if len(implementations) > 1:
            raise MultipleImplementationsError(abstract_type, implementations)

        first_implementation = next(iter(implementations))
        return first_implementation

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        # Remove from nodes
        self._nodes.pop(dependent_type)

        # Remove from type mappings for all base types
        for base_type, implementations in list(self._type_mappings.items()):
            try:
                implementations.remove(dependent_type)
                # Remove empty mappings
                if not implementations:
                    del self._type_mappings[base_type]
            except ValueError:
                continue

        # Remove from resolved instances
        self._resolved_instances.pop(dependent_type, None)

    def get_dependent_types(
        self, dependency_type: NodeDependent
    ) -> list[NodeDependent]:
        """
        Get all types that depend on the given type.
        """
        dependents: list[NodeDependent] = []
        for node_type, node in self._nodes.items():
            for dep_param in node.dependency_params:
                dependent: AbstractDependent[ty.Any] = dep_param.dependency.dependent
                if dependent.dependent_type == dependency_type:
                    dependents.append(node_type)
        return dependents

    def get_dependency_types(self, dependent_type: NodeDependent) -> list[type]:
        """
        Get all types that the given type depends on.
        """
        node = self._nodes[dependent_type]
        deps: list[type] = []
        for param in node.dependency_params:
            dependent_type = param.dependent_type
            deps.append(dependent_type)
        return deps

    def get_initialization_order(self) -> list[type]:
        """
        Return types in dependency order (topological sort).
        """
        order: list[type] = []
        visited = set[type]()

        def visit(node_type: type):
            if node_type in visited:
                return

            visited.add(node_type)
            node = self._nodes[node_type]
            for dep_param in node.dependency_params:
                dependent_type = dep_param.dependent_type
                visit(dependent_type)

            order.append(node_type)

        for node_type in self._nodes:
            visit(node_type)

        return order

    def reset(self) -> None:
        """
        Clear all resolved instances while maintaining registrations.
        """
        self._resolved_instances.clear()

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
            if is_closable(instance):
                await instance.close()
        self.reset()

    def is_resource(self, instance: ty.Any) -> bool:
        return is_closable(instance)

    def is_factory_override(
        self,
        factory_return_type: type[ty.Any],
        factory_or_class: ty.Callable[..., ty.Any] | type[ty.Any],
    ) -> bool:
        """
        Check if a factory or class is overriding an existing node.
        If a we receive a factory with a return type X, we check if X is already registered in the graph.
        """
        is_factory = callable(factory_or_class) and not inspect.isclass(
            factory_or_class
        )
        has_return_type = factory_return_type is not inspect.Signature.empty
        is_registered_node = factory_return_type in self._nodes
        return is_factory and has_return_type and is_registered_node

    def replace_node(
        self, old_node: DependentNode[ty.Any], new_node: DependentNode[ty.Any]
    ) -> None:
        """
        Replace an existing node with a new node.
        """
        self.remove_node(old_node)
        self.register_node(new_node)

    def detect_circular_dependencies(self, node: DependentNode[ty.Any]) -> None:
        """
        Detect circular dependencies.
        """
        raise NotImplementedError

    def resolve(self, dependency_type: NodeDependent, **overrides: ty.Any) -> ty.Any:
        """
        Resolve a dependency and its complete dependency graph.
        Supports dependency overrides for testing.
        Overrides are only applied to the requested type, not its dependencies.
        """

        if is_builtin_type(dependency_type):
            raise TopLevelBulitinTypeError(dependency_type)

        if dependency_type in self._resolved_instances:
            return self._resolved_instances[dependency_type]

        concrete_type = self._resolve_concrete_type(dependency_type)

        try:
            node = self._nodes[concrete_type]
        except KeyError:
            raise UnregisteredTypeError(concrete_type)

        resolved_deps = overrides.copy()

        node.update_forward_dependency_params()

        # Get resolution info for all dependencies
        for dep_param in node.dependency_params:
            resolved_node = dep_param.dependency
            resolved_type = resolved_node.dependent.dependent_type
            if dep_param.name in resolved_deps or dep_param.is_builtin:
                continue

            # Register forward dependencies if needed
            if resolved_type not in self._nodes:
                self.register_node(resolved_node)

            # Resolve dependency if not already resolved
            if resolved_type not in self._resolved_instances:
                self._resolved_instances[resolved_type] = self.resolve(resolved_type)
            resolved_deps[dep_param.name] = self._resolved_instances[resolved_type]

        instance = node.build(**resolved_deps)
        self._resolved_instances[dependency_type] = instance
        return instance

    def register_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """
        if isinstance(node.dependent, ForwardDependent):
            return

        dependent_type: type = (
            ty.get_origin(node.dependent.dependent_type)
            or node.dependent.dependent_type
        )

        if dependent_type in self._nodes:
            return  # Skip if already registered

        if is_builtin_type(dependent_type):
            return

        # Register main type
        self._nodes[dependent_type] = node

        # Update type mappings for dependency resolution
        self._type_mappings[dependent_type].append(dependent_type)

        # __mro__[1:-1] skip dependent itself and object
        for base in dependent_type.__mro__[1:-1]:
            self._type_mappings[base].append(dependent_type)

        # Recursively register dependencies first
        for dep_param in node.dependency_params:
            if not is_builtin_type(dep_param.dependency.dependent):
                self.register_node(dep_param.dependency)

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
        ### Decorator to register a node in the dependency graph.

        - Can be used with both factory functions and classes.
        - If decorating a factory function that returns an existing type,
        it will override the factory for that type.
        ----
        Examples:

        - Register a class:

        ```python
        dg = DependencyGraph()

        @dg.node
        class AuthService: ...
        ```

        - Register a factory function:

        ```python
        @dg.node
        def auth_service_factory() -> AuthService: ...
        ```
        """

        return_type = get_full_typed_signature(factory_or_class).return_annotation
        if self.is_factory_override(return_type, factory_or_class):
            # Create new node with the factory
            node = DependentNode.from_node(factory_or_class)
            old_node = self._nodes[return_type]
            self.replace_node(old_node, node)
        else:
            node = DependentNode.from_node(factory_or_class)
            self.register_node(node)
        return factory_or_class
