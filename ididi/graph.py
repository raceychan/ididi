import inspect
import typing as ty
from collections import defaultdict
from types import MappingProxyType

from .errors import (
    CircularDependencyDetectedError,
    MissingImplementationError,
    MultipleImplementationsError,
    TopLevelBulitinTypeError,
    UnregisteredTypeError,
)
from .node import DependentNode, ForwardDependent, ForwardDependentNode
from .types import (
    GraphNodes,
    GraphNodesView,
    NodeDependent,
    ResolvedInstances,
    TypeMappings,
    TypeMappingView,
)
from .utils.typing_utils import get_full_typed_signature, is_builtin_type


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
    _resolved_instances: ResolvedInstances[ty.Any]
    _type_mappings: TypeMappings[ty.Any]

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
    def type_mappings(self) -> TypeMappingView[ty.Any]:
        return MappingProxyType(self._type_mappings)

    def _resolve_concrete_type(self, abstract_type: type) -> type:
        """
        Resolve abstract type to concrete implementation.
        """
        # If the type is already concrete and registered, return it
        if abstract_type in self._nodes:
            return abstract_type

        implementations: list[type | ForwardDependent] = self._type_mappings.get(
            abstract_type, []
        )

        if not implementations:
            raise MissingImplementationError(abstract_type)
        if len(implementations) > 1:
            raise MultipleImplementationsError(abstract_type, implementations)

        first_implementation = ty.cast(type, next(iter(implementations)))
        return first_implementation

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent

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
        self, dependency_type: NodeDependent[ty.Any]
    ) -> list[NodeDependent[ty.Any]]:
        """
        Get all types that depend on the given type.
        """
        dependents: list[NodeDependent[ty.Any]] = []
        for node_type, node in self._nodes.items():
            for dep_param in node.dependency_params:
                if dep_param.dependency.dependent == dependency_type:
                    dependents.append(node_type)
        return dependents

    def get_dependency_types(self, dependent_type: NodeDependent[ty.Any]) -> list[type]:
        """
        Get all types that the given type depends on.
        """
        node = self._nodes[dependent_type]
        deps: list[type] = []
        for param in node.dependency_params:
            dependent_type = param.dependency.dependent
            if isinstance(dependent_type, ForwardDependent):
                deps.append(dependent_type.resolve())
            else:
                deps.append(dependent_type)
        return deps

    def get_initialization_order(self) -> list[NodeDependent[ty.Any]]:
        """
        Return types in dependency order (topological sort).
        """
        order: list[NodeDependent[ty.Any]] = []
        visited = set[NodeDependent[ty.Any]]()

        def visit(node_type: NodeDependent[ty.Any]):
            if node_type in visited:
                return

            visited.add(node_type)
            node = self._nodes[node_type]
            for dep_param in node.dependency_params:
                visit(dep_param.dependency.dependent)

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
            if self.is_resource(instance):
                await instance.close()
        self.reset()

    def is_resource(self, type_: ty.Any) -> bool:
        # TODO: add more resource types
        return hasattr(type_, "close") and callable(type_.close)

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

    def resolve[I](self, dependency_type: type[I], **overrides: ty.Any) -> I:
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

        for dep_param in node.dependency_params:
            param = dep_param.param
            dep = dep_param.dependency.dependent

            if isinstance(dep, ForwardDependent):
                resolved_type: type = dep.resolve()
                sub_params = get_full_typed_signature(resolved_type).parameters.values()
                for sub_param in sub_params:
                    if sub_param.annotation is concrete_type:
                        raise CircularDependencyDetectedError(
                            [dependency_type, resolved_type]
                        )

                resolved_node = DependentNode.from_node(resolved_type)

                if resolved_type not in self._nodes:
                    self.register_node(resolved_node)
            else:
                resolved_type: type = dep

            if param.name in resolved_deps or is_builtin_type(resolved_type):
                continue

            if resolved_type not in self._resolved_instances:
                self._resolved_instances[resolved_type] = self.resolve(resolved_type)
            resolved_deps[param.name] = self._resolved_instances[resolved_type]

        instance = node.build(**resolved_deps)
        self._resolved_instances[dependency_type] = instance
        return instance

    def register_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """
        if node.dependent in self._nodes:
            return  # Skip if already registered

        if not isinstance(node.dependent, ForwardDependent):
            # Register main type
            self._nodes[node.dependent] = node
            # Update type mappings for dependency resolution
            self._type_mappings[node.dependent].append(node.dependent)
            # __mro__[1:-1] skip dependent itself and object
            for base in node.dependent.__mro__[1:-1]:
                self._type_mappings[base].append(node.dependent)

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
        Decorator to register a node in the dependency graph.
        Can be used with both factory functions and classes.
        If decorating a factory function that returns an existing type,
        it will override the factory for that type.
        """
        return_type = get_full_typed_signature(factory_or_class).return_annotation
        # Handle factory override
        if self.is_factory_override(return_type, factory_or_class):
            # Create new node with the factory
            node = DependentNode.from_node(factory_or_class)
            old_node = self._nodes[return_type]
            self.replace_node(old_node, node)
        else:
            node = DependentNode.from_node(factory_or_class)
            self.register_node(node)
        return factory_or_class
