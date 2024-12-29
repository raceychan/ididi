from typing import Any, Callable, ForwardRef, Hashable, TypeVar, Union


class IDIDIError(Exception):
    """
    Base class for all IDIDI exceptions.
    """

    def __init__(self, message: str, /):
        self.message = message
        super().__init__(self.message)


# =============== General Errors ===============


class NotSupportedError(IDIDIError):
    """
    Base class for all not supported exceptions.
    """


# =============== Node Errors ===============


class NodeError(IDIDIError):
    """
    Base class for all node related exceptions.
    """


class NodeResolveError(IDIDIError):
    """
    Base class for all node resolve related exceptions.
    """


class AsyncResourceInSyncError(NodeResolveError):
    def __init__(self, factory: Any):
        self.factory = factory
        super().__init__(
            f"Requiring async resource {factory} in a sync scope is not supported"
        )


class ResourceOutsideScopeError(NodeResolveError):
    def __init__(self, dependent: type):
        super().__init__(f"resource {dependent} has to be resolved within scope")


class PositionalOverrideError(NodeResolveError):
    """
    Raised when a positional override is used.
    """

    def __init__(self, args: Any):
        super().__init__(
            f"Positional overrides {args} are not supported, use keyword arguments instead"
        )


class UnsolvableNodeError(NodeResolveError):
    def __init__(self, message: str):
        super().__init__(message)
        self.__notes__: list[str] = []

    def add_note(self, note: str) -> None:
        self.__notes__.append(note)

    def add_context(self, dependent: type, param_name: str, param_annotation: type):
        msg = f"<- {dependent.__name__}({param_name}: {param_annotation.__name__})"
        self.add_note(msg)


class ProtocolFacotryNotProvidedError(UnsolvableNodeError):
    """
    Raised when a protocol is used as a dependency without a factory.
    """

    def __init__(self, protocol: type):
        super().__init__(
            f"Protocol {protocol} can't be instantiated, a factory is required to resolve it"
        )


class ABCNotImplementedError(UnsolvableNodeError):
    """
    Raised when an ABC is used as a dependency without a factory.
    """

    def __init__(self, abc: type, abstract_methods: frozenset[str]):
        super().__init__(
            f"ABC {abc} has no valid implementations, either provide a implementation that implements {abstract_methods} or a factory"
        )


class UnsolvableParameterError(UnsolvableNodeError):
    """
    Raised when a parameter is unsolveable.
    """


class UnsolvableDependencyError(UnsolvableParameterError):
    """
    Raised when a dependency parameter can't be built.
    """

    def __init__(
        self,
        *,
        dep_name: str,
        factory: Union[Callable[..., Any], type],
        dependent_type: type,
        dependency_type: type,
    ):
        type_repr = getattr(dependency_type, "__name__", str(dependency_type))
        param_repr = f" * {dependent_type.__name__}({dep_name}: {type_repr}), value of `{dep_name}` must be provided"
        self.message = (
            f"Unable to resolve dependency for parameter in {factory}, \n{param_repr}"
        )
        super().__init__(self.message)


class ForwardReferenceNotFoundError(UnsolvableParameterError):
    """
    Raised when a forward reference can't be found in the global namespace.
    """

    def __init__(self, forward_ref: ForwardRef):
        msg = f"Unable to resolve forward reference: {forward_ref}, make sure it has been defined in the global namespace"
        super().__init__(msg)


class MissingAnnotationError(UnsolvableParameterError):
    def __init__(
        self,
        *,
        dependent_type: type,
        param_name: str,
        param_type: type,
    ):
        self.param_name = param_name
        self.dependent = param_type

        param_repr = f" * value of `{param_name}` must be provided"
        message = f"Unable to resolve dependency for parameter in {dependent_type}, \n{param_repr}"
        super().__init__(message)


class UnsolvableReturnTypeError(UnsolvableParameterError):
    """
    Raised when a factory has no return type.
    Thus can't be determined what it is trying to override
    """

    def __init__(self, factory: Callable[..., Any], target: type):
        self.factory = factory
        msg = f"Factory {factory} must have a return type, instead of {target}"
        super().__init__(msg)


class GenericDependencyNotSupportedError(NodeError):
    """
    Raised when attempting to use a generic type that is not yet supported.
    """

    def __init__(self, generic_type: Union[type, TypeVar]):
        super().__init__(
            f"Using generic a type as a dependency is not yet supported: {generic_type}"
        )


# =============== Graph Errors ===============
class GraphError(IDIDIError):
    """
    Base class for all graph related exceptions.
    """


class OutOfScopeError(GraphError):
    def __init__(self, name: Hashable = ""):
        if name:
            msg = f"scope with {name=} not found in current context"
        else:
            msg = "scope not found in current context"

        super().__init__(msg)


class MergeWithScopeStartedError(GraphError):
    def __init__(self):
        super().__init__("Merging graph within scope is not supported")


class GraphResolveError(GraphError):
    """
    Base class for all graph resolving related exceptions.
    """


class CircularDependencyDetectedError(GraphResolveError):
    """Raised when a circular dependency is detected in the dependency graph."""

    def __init__(self, cycle_path: list[type]):
        cycle_str = " -> ".join(t.__name__ for t in cycle_path)
        self._cycle_path = cycle_path
        super().__init__(f"Circular dependency detected: {cycle_str}")

    @property
    def cycle_path(self) -> list[type]:
        return self._cycle_path


class ReusabilityConflictError(GraphResolveError):
    def __init__(self, path: list[type], nonreuse: type):
        conflict_str = " -> ".join(t.__name__ for t in path)
        msg = f"""Transient dependency `{nonreuse.__name__}` with reuse dependents \
        \n make sure each of {conflict_str} is configured as `reuse=False` \
        """
        super().__init__(msg)


class TopLevelBulitinTypeError(GraphResolveError):
    """
    Raised when a builtin type is used as a top level dependency.
    Example:
    >>> dag.resolve(int)
    """

    def __init__(self, dependency_type: Union[type, Callable[..., Any]]):
        super().__init__(
            f"Using builtin type {dependency_type} as a top level dependency is not supported"
        )


class MissingImplementationError(GraphResolveError):
    """
    Raised when a type has no implementations.
    """

    def __init__(self, dependency_type: type):
        super().__init__(f"No implementations found for {dependency_type}")
