import functools
import inspect
import typing as ty
from collections import defaultdict

# from contextlib import AsyncExitStack, ExitStack
from types import MappingProxyType, TracebackType

from .errors import (
    MissingImplementationError,
    MultipleImplementationsError,
    TopLevelBulitinTypeError,
)
from .node import AbstractDependent, DependentNode, ForwardDependent
from .types import INodeConfig, NodeConfig, T_Factory, TDecor
from .utils.typing_utils import get_full_typed_signature, is_builtin_type, is_closable

type NodeDependent[T] = type[T]
"""
### A dependent can be a concrete type or a forward reference
"""

type GraphNodes[I] = dict[type[I], DependentNode[I]]
"""
### mapping a type to its corresponding node
"""

type GraphNodesView[I] = MappingProxyType[type[I], DependentNode[I]]
"""
### a readonly view of GraphNodes
"""

type ResolvedInstances[T] = dict[type[T], T]
"""
mapping a type to its resolved instance
"""

type TypeMappings[T] = dict[NodeDependent[T], list[NodeDependent[T]]]
"""
### mapping a type to its dependencies
"""


class ImplemntationRegistry:
    """
    A mapping of dependent types to their implementations.
    """

    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: TypeMappings[ty.Any] = defaultdict(list)

    def __getitem__[T](self, dependent_type: type[T]) -> list[type[T]]:
        return self._mappings[dependent_type].copy()

    def __contains__(self, dependent_type: type) -> bool:
        return dependent_type in self._mappings

    def register(self, dependent_type: type):
        """
        Register a dependent type and update its base types.
        """
        self._mappings[dependent_type].append(dependent_type)

        # __mro__[1:-1] skip dependent itself and object
        for base in dependent_type.__mro__[1:-1]:
            self._mappings[base].append(dependent_type)

    def remove(self, dependent_type: type):
        """
        Remove a dependent type and update its base types.
        """

        for base in dependent_type.__mro__[1:-1]:
            self._mappings[base].remove(dependent_type)

        del self._mappings[dependent_type]

    def get(
        self, dependent_type: type, /, default: list[type] | None = None
    ) -> list[type] | None:
        return self._mappings.get(dependent_type, default)


class ResolutionRegistry:
    """
    A mapping of dependent types to their resolved instances.
    """

    __slots__ = ("_mappings",)

    def __init__(self):
        self._mappings: ResolvedInstances[ty.Any] = {}

    def __len__(self) -> int:
        return len(self._mappings)

    def __contains__(self, dependent_type: type) -> bool:
        return dependent_type in self._mappings

    def __getitem__[T](self, dependent_type: type[T]) -> T:
        return self._mappings[dependent_type]

    def remove(self, dependent_type: type) -> None:
        self._mappings.pop(dependent_type, None)

    def clear(self) -> None:
        self._mappings.clear()

    def register(self, dependent_type: type, instance: ty.Any) -> None:
        self._mappings[dependent_type] = instance

    async def close(self, dependent_type: type) -> None:
        instance = self._mappings.pop(dependent_type, None)
        if instance and is_closable(instance):
            await instance.close()

    async def close_all(self, close_order: ty.Iterable[type]) -> None:
        for type_ in close_order:
            await self.close(type_)


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

    def __init__(self):
        self._nodes: GraphNodes[ty.Any] = {}
        # self._resolved_instances: ResolvedInstances = {}
        self._resolution_registry = ResolutionRegistry()
        self._type_registry = ImplemntationRegistry()

        # Set of resolved nodes, to avoid resolving the same node multiple times
        self._resolved_nodes: set[DependentNode[ty.Any]] = set()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"nodes={len(self._nodes)}, "
            f"resolved={len(self._resolution_registry)})"
        )

    def __stats__(self) -> str:
        """
        Returns a concise statistics of the graph showing:
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
            f"resolved={len(self._resolution_registry)}, "
            f"roots=[{', '.join(t.__name__ for t in sorted(roots, key=lambda x: x.__name__))}], "
            f"leaves=[{', '.join(t.__name__ for t in sorted(leaves, key=lambda x: x.__name__))}])"
        )

    async def __aenter__(self) -> "DependencyGraph":
        """
        should we reopen closed resources when entering a graph?
        """
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_value: ty.Any,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def nodes(self) -> GraphNodesView[ty.Any]:
        return MappingProxyType(self._nodes)

    @property
    def type_mappings(self) -> ImplemntationRegistry:
        return self._type_registry

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.discard(node)

    def get_dependent_types[
        T
    ](self, dependency_type: NodeDependent[T]) -> list[NodeDependent[T]]:
        """
        Get all types that depend on the given type.
        """
        dependents: list[NodeDependent[T]] = []
        for node_type, node in self._nodes.items():
            for dep_param in node.dependency_params:
                dependent: AbstractDependent[ty.Any] = dep_param.dependency.dependent
                if dependent.dependent_type == dependency_type:
                    dependents.append(node_type)
        return dependents

    def get_dependency_types[
        T
    ](self, dependent_type: NodeDependent[T]) -> list[type[T]]:
        """
        Get all types that the given type depends on.
        """
        node = self._nodes[dependent_type]
        deps: list[type[T]] = []
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
        self._resolution_registry.clear()

    async def aclose(self) -> None:
        """
        Close any resources held by resolved instances.
        Closes in reverse initialization order.
        """
        close_order = reversed(self.get_initialization_order())
        await self._resolution_registry.close_all(close_order)
        self.reset()

    def is_factory_override[
        **P
    ](
        self,
        factory_return_type: type[ty.Any],
        factory_or_class: T_Factory[ty.Any, P],
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

    def _resolve_concrete_type(self, abstract_type: type) -> type:
        """
        Resolve abstract type to concrete implementation.
        """
        # If the type is already concrete and registered, return it
        if abstract_type in self._nodes:
            return abstract_type

        implementations: list[type] | None = self._type_registry.get(abstract_type)

        if not implementations:
            raise MissingImplementationError(abstract_type)

        if len(implementations) > 1:
            raise MultipleImplementationsError(abstract_type, implementations)

        first_implementations = implementations[0]
        return first_implementations

    def resolve_node[T](self, dependency_type: type[T]) -> DependentNode[T]:
        """
        Resolve which node should be used for the given type and its dependencies.
        Returns the resolved node and a mapping of parameter names to their dependency types.
        """
        if is_builtin_type(dependency_type):
            raise TopLevelBulitinTypeError(dependency_type)

        # Resolve concrete implementation
        try:
            concrete_type = self._resolve_concrete_type(dependency_type)
        except MissingImplementationError:
            node = DependentNode.from_node(dependency_type, NodeConfig())
            self.register_node(node)
        else:
            node = self._nodes[concrete_type]

        # Handle forward dependencies
        if node not in self._resolved_nodes:
            for actualized_node in node.actualize_forward_deps():
                self.register_node(actualized_node)
            self._resolved_nodes.add(node)
        return node

    def resolve[
        T, **P
    ](self, dependency_type: T_Factory[T, P], /, **overrides: ty.Any) -> T:
        """
        Resolve a dependency and bild its complete dependency graph.
        Supports dependency overrides for testing.
        Overrides are only applied to the requested type, not its dependencies.
        """
        node_dep_type = ty.cast(type[T], dependency_type)

        if node_dep_type in self._resolution_registry:
            return self._resolution_registry[node_dep_type]

        node: DependentNode[T] = self.resolve_node(node_dep_type)
        resolved_deps = overrides.copy()

        # Resolve and build all dependencies
        for dep_param in node.dependency_params:
            sub_node = dep_param.dependency
            param_name = dep_param.name
            dep_type = sub_node.dependent_type
            if param_name in resolved_deps or dep_param.is_builtin:
                continue

            resolved_dep = self.resolve(dep_type)
            resolved_deps[param_name] = resolved_dep

        instance = node.build(**resolved_deps)
        if node.config.reuse:
            self._resolution_registry.register(node_dep_type, instance)
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

        self._type_registry.register(dependent_type)

    @ty.overload
    def node[
        I
    ](self, factory_or_class: type[I], /, **config: ty.Unpack[INodeConfig]) -> type[
        I
    ]: ...

    @ty.overload
    def node[
        I, **P
    ](
        self, factory_or_class: ty.Callable[P, I], /, **config: ty.Unpack[INodeConfig]
    ) -> ty.Callable[P, I]: ...

    @ty.overload
    def node[
        I, **P
    ](self, **config: ty.Unpack[INodeConfig]) -> TDecor[T_Factory[I, P]]: ...

    def node[
        I, **P
    ](
        self,
        factory_or_class: T_Factory[I, P] | None = None,
        **config: ty.Unpack[INodeConfig],
    ) -> (T_Factory[I, P] | TDecor[T_Factory[I, P]]):
        """
        ### Decorator to register a node in the dependency graph.

        - Can be used with both factory functions and classes.
        - If decorating a factory function that returns an existing type,
        it will override the factory for that type.
        ----
        Examples:

        #### Register a class:

        ```python
        dg = DependencyGraph()

        @dg.node
        class AuthService: ...
        ```

        #### Register a factory function:

        ```python
        @dg.node
        def auth_service_factory() -> AuthService: ...
        ```
        """
        if not factory_or_class:
            return ty.cast(
                TDecor[T_Factory[I, P]],
                functools.partial(self.node, **config),
            )

        node_config = NodeConfig(**config)

        return_type = get_full_typed_signature(factory_or_class).return_annotation
        if self.is_factory_override(return_type, factory_or_class):
            new_node = DependentNode.from_node(factory_or_class, node_config)
            old_node = self._nodes[return_type]
            self.replace_node(old_node, new_node)
        else:
            node = DependentNode.from_node(factory_or_class, node_config)
            self.register_node(node)
        return factory_or_class
