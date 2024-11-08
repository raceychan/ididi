import inspect
import typing as ty
from functools import lru_cache, partial
from types import MappingProxyType, TracebackType

from .errors import MissingImplementationError, TopLevelBulitinTypeError
from .node import AbstractDependent, DependentNode
from .registry import GraphNodes, GraphNodesView, ResolutionRegistry, TypeRegistry
from .types import INSPECT_EMPTY, GraphConfig, IFactory, INodeConfig, NodeConfig, TDecor
from .utils.typing_utils import get_full_typed_signature, is_builtin_type, is_function


class DependencyGraph:
    """
    ### Description:
    A dependency DAG (Directed Acyclic Graph) that manages dependency nodes and their relationships.

    [config]
    static_resolve: bool
    ---
    whether to statically resolve all nodes when entering the graph.
    """

    def __init__(self, static_resolve: bool = True):
        self._nodes: GraphNodes[ty.Any] = {}
        self._resolved_nodes: GraphNodes[ty.Any] = {}
        self._resolution_registry = ResolutionRegistry()
        self._type_registry = TypeRegistry()
        self._config = GraphConfig(static_resolve=static_resolve)

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
        if not self._config.static_resolve:
            return self

        for node_type in self._nodes.copy():
            if node_type in self._resolved_nodes:
                continue
            self.static_resolve(node_type)
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
    def resolution_registry(self) -> ResolutionRegistry:
        """
        A mapping of dependent types to their resolved instances.
        """
        return self._resolution_registry

    @property
    def type_registry(self) -> TypeRegistry:
        """
        A mapping of abstract types to their implementations.
        """
        return self._type_registry

    @property
    def resolved_nodes(self) -> GraphNodesView[ty.Any]:
        """
        A node is considered resolved if all its dependencies have been resolved.
        Which means all its forward dependencies have been resolved.
        This would also include builtin types.
        """
        return MappingProxyType(self._resolved_nodes)

    def remove_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Remove a node from the graph and clean up all its references.
        """
        dependent_type = node.dependent.dependent_type

        self._nodes.pop(dependent_type)
        self._type_registry.remove(dependent_type)
        self._resolution_registry.remove(dependent_type)
        self._resolved_nodes.pop(dependent_type, None)

    def get_dependent_types[T](self, dependent: type[T]) -> list[type[T]]:
        """
        Get all types that depend on the given type.
        """
        dependents: list[type[T]] = []
        for node_type, node in self._nodes.items():
            for dep_param in node.signature:
                sub_dependent: AbstractDependent[ty.Any] = dep_param.node.dependent
                if sub_dependent.dependent_type == dependent:
                    dependents.append(node_type)
        return dependents

    def get_dependency_types[T](self, dependent: type[T]) -> list[type[T]]:
        """
        Get all types that the given type depends on.
        """
        node = self._nodes[dependent]
        deps: list[type[T]] = []
        for param in node.signature:
            dependent = param.dependent_type
            deps.append(dependent)
        return deps

    def top_sorted_dependencies(self) -> list[type]:
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
            for dep_param in node.signature:
                dependent_type = dep_param.dependent_type
                if dependent_type not in self._nodes:
                    continue
                visit(dependent_type)

            order.append(node_type)

        for node_type in self._nodes:
            visit(node_type)

        return order

    def reset(self, clear_resolved: bool = False) -> None:
        """
        Clear all resolved instances while maintaining registrations.

        clear_resolved: bool
        ---
        whether to clear:
        - the node registry,
        - the type registry,
        """
        self._resolution_registry.clear()
        if clear_resolved:
            self._type_registry.clear()
            self._resolved_nodes.clear()
            self._nodes.clear()

    async def aclose(self) -> None:
        """
        Close any resources held by resolved instances.
        Closes in reverse initialization order.
        """
        close_order = reversed(self.top_sorted_dependencies())
        await self._resolution_registry.close_all(close_order)
        self.reset(clear_resolved=True)

    def _resolve_concrete_type[T](self, abstract_type: type[T]) -> type[T]:
        """
        Resolve abstract type to concrete implementation.
        """
        implementations: list[type] | None = self._type_registry.get(abstract_type)

        if not implementations:
            raise MissingImplementationError(abstract_type)

        last_implementations = implementations[-1]
        return last_implementations

    def _resolve_concrete_node[T](self, dependent: type[T]) -> DependentNode[T]:
        if registered_node := self._nodes.get(dependent):
            if is_function(registered_node.factory):
                return registered_node

        try:
            concrete_type = self._resolve_concrete_type(dependent)
        except MissingImplementationError:
            node = DependentNode.from_node(dependent, NodeConfig())
            self.register_node(node)
        else:
            node = self._nodes[concrete_type]
        return node

    def replace_node(
        self, old_node: DependentNode[ty.Any], new_node: DependentNode[ty.Any]
    ) -> None:
        """
        Replace an existing node with a new node.
        """
        self.remove_node(old_node)
        self.register_node(new_node)

    def static_resolve[
        T, **P
    ](self, dependent_factory: type[T] | ty.Callable[P, T]) -> DependentNode[T]:
        """
        Resolve a dependency without building its instance.
        """
        if is_function(dependent_factory):
            sig = get_full_typed_signature(dependent_factory, check_return=True)
            dependent: type[T] = sig.return_annotation
        else:
            dependent = ty.cast(type[T], dependent_factory)

        if dependent in self._resolved_nodes:
            return self._resolved_nodes[dependent]

        if is_builtin_type(dependent):
            raise TopLevelBulitinTypeError(dependent)

        node = self._resolve_concrete_node(dependent)

        for subnode in node.iter_dependencies():
            resolved_node = self.static_resolve(subnode.dependent_type)
            self.register_node(resolved_node)
            self._resolved_nodes[subnode.dependent_type] = resolved_node

        node.check_for_resolvability()

        self._resolved_nodes[dependent] = node
        return node

    @ty.overload
    def resolve[**P, T](self, dependent: ty.Callable[P, T], /) -> T: ...

    @ty.overload
    def resolve[
        T, **P
    ](
        self, dependent: ty.Callable[P, T], /, *args: P.args, **overrides: P.kwargs
    ) -> T: ...

    def resolve[
        **P, T
    ](self, dependent: ty.Callable[P, T], /, **overrides: ty.Any) -> T:
        """
        Resolve a dependency and bild its complete dependency graph.
        Supports dependency overrides for testing.
        Overrides are only applied to the requested type, not its dependencies.
        """
        node_dep_type = dependent

        if node_dep_type in self._resolution_registry:
            return self._resolution_registry[node_dep_type]

        node: DependentNode[T] = self.static_resolve(node_dep_type)

        resolved_deps: dict[str, ty.Any] = overrides.copy()

        for dep_param in node.signature:
            dep_name = dep_param.name
            if dep_param.should_not_resolve:
                continue

            if dep_name in overrides:
                continue

            sub_node_dep_type = dep_param.node.dependent_type

            resolved_dep = self.resolve(sub_node_dep_type)
            resolved_deps[dep_name] = resolved_dep

        instance = node.build(**resolved_deps)
        if node.config.reuse:
            self._resolution_registry.register(node_dep_type, instance)

        return ty.cast(T, instance)

    @lru_cache
    def factory[T](self, dependent: type[T]) -> IFactory[T, ...]:
        """
        A helper function that creates a resolver for a given type.
        """
        if dependent not in self._resolved_nodes:
            self.static_resolve(dependent)

        def resolver() -> T:
            return self.resolve(dependent)

        return resolver

    def entry_node[
        **P, R
    ](self, func: ty.Callable[P, R], /, **kwargs: ty.Unpack[INodeConfig]) -> (
        ty.Callable[[], R] | ty.Callable[[], ty.Awaitable[R]]
    ):
        Dummy = type("Dummy", (object,), {})
        dep = ty.cast(type[R], Dummy)
        sig = get_full_typed_signature(func)
        node = DependentNode.create(
            dependent=dep, factory=func, signature=sig, config=NodeConfig(**kwargs)
        )
        self.register_node(node)

        def resolve(*args: P.args, **kwargs: P.kwargs) -> R:
            r = self.resolve(dep, *args, **kwargs)
            return r

        async def async_resolve(*args: P.args, **kwargs: P.kwargs) -> R:
            r = await ty.cast(ty.Awaitable[R], self.resolve(dep, *args, **kwargs))
            return r

        if inspect.iscoroutinefunction(func):
            f = async_resolve
        else:
            f = resolve
        return f

    def register_node(self, node: DependentNode[ty.Any]) -> None:
        """
        Register a dependency node and update dependency relationships.
        Automatically registers any unregistered dependencies.
        """

        dep_type = node.dependent.dependent_type
        dependent_type: type = ty.get_origin(dep_type) or dep_type

        # Skip if registered
        if dependent_type in self._nodes:
            return

        # Register main type
        self._nodes[dependent_type] = node

        # Register type mappings
        self._type_registry.register(dependent_type)

    @ty.overload
    def node[I](self, factory_or_class: type[I]) -> type[I]: ...

    @ty.overload
    def node[I, **P](self, factory_or_class: IFactory[I, P]) -> IFactory[I, P]: ...

    @ty.overload
    def node(self, **config: ty.Unpack[INodeConfig]) -> TDecor: ...

    def node[
        I, **P
    ](
        self,
        factory_or_class: IFactory[I, P] | type[I] | None = None,
        **config: ty.Unpack[INodeConfig],
    ) -> (IFactory[I, P] | type[I] | TDecor):
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
            configed = ty.cast(TDecor, partial(self.node, **config))
            return configed

        node_config = NodeConfig(**config)

        return_type = get_full_typed_signature(factory_or_class).return_annotation

        if return_type is not INSPECT_EMPTY and return_type in self._nodes:
            old_node = self._nodes[return_type]
            self.remove_node(old_node)

        node = DependentNode.from_node(factory_or_class, node_config)
        self.register_node(node)
        return factory_or_class
