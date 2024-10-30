import typing as ty


class IDIDIError(Exception):
    """
    Base class for all IDIDI exceptions.
    """


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


class UnsolvableDependencyError(NodeError):
    def __init__(self, param_name: str, required_type: ty.Any):
        self.param_name = param_name
        self.required_type = required_type
        super().__init__(
            f"Unable to resolve dependency for parameter: {param_name}, value of {required_type} must be provided"
        )


class ForwardReferenceNotFoundError(NodeError):
    """
    Raised when a forward reference can't be found in the global namespace.
    """

    def __init__(self, forward_ref: ty.ForwardRef):
        msg = f"Unable to resolve forward reference: {forward_ref}, make sure it has been defined in the global namespace"
        super().__init__(msg)


class MissingAnnotationError(NodeError):
    def __init__(self, dependent: type, param_name: str):
        self.dependent = dependent
        self.param_name = param_name
        msg = f"Unable to resolve dependency for parameter: {param_name} in {dependent}, annotation for `{param_name}` must be provided"
        super().__init__(msg)


class GenericDependencyNotSupportedError(NodeError):
    """
    Raised when attempting to use a generic type that is not yet supported.
    """

    def __init__(self, generic_type: type | ty.TypeVar):
        super().__init__(
            f"Using generic a type as a dependency is not yet supported: {generic_type}"
        )


class ProtocolFacotryNotProvidedError(NodeError):
    """
    Raised when a protocol is used as a dependency without a factory.
    """

    def __init__(self, protocol: type):
        super().__init__(
            f"Protocol {protocol} can't be instantiated, a factory is required to resolve it"
        )


class ABCWithoutImplementationError(NodeError):
    """
    Raised when an ABC is used as a dependency without a factory.
    """

    def __init__(self, abc: type, abstract_methods: frozenset[str]):
        super().__init__(
            f"ABC {abc} has no valid implementations, either provide a implementation that implements {abstract_methods} or a factory"
        )


# =============== Graph Errors ===============
class GraphError(IDIDIError):
    """
    Base class for all graph related exceptions.
    """


class CircularDependencyDetectedError(GraphError):
    """Raised when a circular dependency is detected in the dependency graph."""

    def __init__(self, cycle_path: list[type]):
        cycle_str = " -> ".join(t.__name__ for t in cycle_path)
        self._cycle_path = cycle_path
        super().__init__(f"Circular dependency detected: {cycle_str}")

    @property
    def cycle_path(self) -> list[type]:
        return self._cycle_path


class TopLevelBulitinTypeError(GraphError):
    """
    Raised when a builtin type is used as a top level dependency.
    Example:
    >>> dag.resolve(int)
    """

    def __init__(self, dependency_type: type):
        super().__init__(
            f"Using builtin type {dependency_type} as a top level dependency is not supported"
        )


class MissingImplementationError(GraphError):
    """
    Raised when a type has no implementations.
    """

    def __init__(self, dependency_type: type):
        super().__init__(f"No implementations found for {dependency_type}")


class MultipleImplementationsError(GraphError):
    """
    Raised when a type has multiple implementations.
    """

    def __init__(self, dependency_type: type, implementations: ty.Iterable[ty.Any]):
        implementations_str = ", ".join(t.__name__ for t in implementations)
        super().__init__(
            f"Multiple implementations found for {dependency_type}: {implementations_str}"
        )
