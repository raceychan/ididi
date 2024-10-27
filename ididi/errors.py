import typing as ty


class IDIDIError(Exception):
    """
    Base class for all IDIDI exceptions.
    """


class GraphError(IDIDIError):
    """
    Base class for all graph related exceptions.
    """


class CircularDependencyDetectedError(GraphError):
    """Raised when a circular dependency is detected in the dependency graph."""

    def __init__(self, cycle_path: list[type] ):
        cycle_str = " -> ".join(t.__name__ for t in cycle_path)
        super().__init__(f"Circular dependency detected: {cycle_str}")


class UnsolvableDependencyError(IDIDIError):
    def __init__(self, param_name: str, required_type: type):
        self.param_name = param_name
        self.required_type = required_type
        super().__init__(
            f"Unable to resolve dependency for parameter: {param_name}, value of {required_type} must be provided"
        )


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


class UnregisteredTypeError(GraphError):
    """
    Raised when a type is not registered in the graph.
    """

    def __init__(self, dependency_type: type):
        super().__init__(f"No registration found for {dependency_type}")


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

    def __init__(self, dependency_type: type, implementations: ty.Iterable[type]):
        implementations_str = ", ".join(t.__name__ for t in implementations)
        super().__init__(
            f"Multiple implementations found for {dependency_type}: {implementations_str}"
        )
