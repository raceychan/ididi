import typing as ty


class CircularDependencyError(Exception):
    def __init__(self, dep: type, stack: ty.Iterable[type]):
        cycle = " -> ".join(t.__name__ for t in stack)
        cycle += f" -> {dep.__name__}"
        msg = f"Circular dependency detected: {cycle}"
        super().__init__(msg)


class UnsolvableDependencyError(Exception):
    def __init__(self, param_name: str, required_type: type):
        self.param_name = param_name
        self.required_type = required_type
        super().__init__(
            f"Unable to resolve dependency for parameter: {param_name}, value of {required_type} must be provided"
        )


class TopLevelBulitinTypeError(Exception):
    """
    Raised when a builtin type is used as a top level dependency.
    Example:
    >>> dag.resolve(int)
    """

    def __init__(self, dependency_type: type):
        super().__init__(
            f"Using builtin type {dependency_type} as a top level dependency is not supported"
        )
