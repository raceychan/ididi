from typing import Any, Callable, ForwardRef, Hashable, Union


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

class DeprecatedError(IDIDIError):
    """
    Base class for all deprecated error
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
    def __init__(self, dependent: Any):
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

    def add_context(
        self,
        dependent: Callable[..., Any],
        param_name: str,
        param_annotation: Callable[..., Any],
    ):
        dep_repr = getattr(dependent, "__name__", str(dependent))
        param_repr = getattr(param_annotation, "__name__", str(param_annotation))
        msg = f"-> {dep_repr}({param_name}: {param_repr})"
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
            f"ABC {abc} has no valid implementations, either provide a implementation that implements {tuple(m for m in abstract_methods)} or a factory"
        )


class UnsolvableParameterError(UnsolvableNodeError):
    """
    Raised when a parameter is unsolveable.
    """

class ConfigConflictError(UnsolvableNodeError):
    """
    Raised when two nodes with same resolve order 
    but different config
    """

    def __init__(self, current_factory: Any, current_reuse: bool, incoming_factory: Any, incoming_reuse: bool):
        self.curren_factory = current_factory
        self.current_reuse = current_reuse
        self.incoming_factory = incoming_factory
        self.incoming_reuse = incoming_reuse

        msg = f"""
        Dependent has conflict in reusability, current factory {self.curren_factory} is set to reuse={self.current_reuse}, \n\t whereas incoming factory{self.incoming_factory} is set to reuse={self.incoming_reuse}
        """.strip()
        super().__init__(msg)

class ParamReusabilityConflictError(UnsolvableNodeError):
    def __init__(self, dependent_qualname: str, param_name: str, cce: ConfigConflictError):
        msg = f"""\n\t Dependent {dependent_qualname!r} has conflict in param: \n\t {param_name!r} with current factory {cce.curren_factory} is set to reuse={cce.current_reuse}, \n\t whereas incoming factory {cce.incoming_factory} is set to reuse={cce.incoming_reuse}
        """.lstrip()
        super().__init__(msg)


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
        dependent_type: Union[type, Callable[..., Any]],
        param_name: str,
        param_type: type,
    ):
        self.param_name = param_name
        self.dependent = param_type

        param_repr = f"\n * value of `{param_name}` must be provided"
        message = f"Unable to resolve dependency for parameter in {dependent_type}: {param_repr}"
        super().__init__(message)


class UnsolvableReturnTypeError(UnsolvableParameterError):
    """
    Raised when a factory has no return type.
    Thus can't be determined what it is trying to override
    """

    def __init__(self, factory: Callable[..., Any], target: type):
        self.factory = factory
        msg = f"Factory {factory} must have a solvable return type, found {target}"
        super().__init__(msg)


# =============== Graph Errors ===============
class GraphError(IDIDIError):
    """
    Base class for all graph related exceptions.
    """


class OutOfScopeError(GraphError):
    def __init__(self, name: Hashable = ""):
        msg = f"scope with {name=} not found in current context"
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

    def __init__(self, cycle_path: list[Callable[..., Any]]):
        cycle_str = " -> ".join(t.__name__ for t in cycle_path)
        self._cycle_path = cycle_path
        super().__init__(f"Circular dependency detected: {cycle_str}")

    @property
    def cycle_path(self) -> list[Callable[..., Any]]:
        return self._cycle_path


class ReusabilityConflictError(GraphResolveError):
    def __init__(self, path: list[Callable[..., Any]], nonreuse: Any):
        conflict_str = " -> ".join(t.__name__ for t in path)
        msg = f"""Transient dependency `{nonreuse.__name__}` with reuse dependents \
        \n make sure each of its dependents {conflict_str} is configured as `reuse=False` \
        """
        super().__init__(msg)


class TopLevelBulitinTypeError(GraphResolveError):
    """
    Raised when a builtin type is used as a top level dependency.
    Example:
    >>> dag.resolve(int)
    """

    def __init__(self, dependency_type: Any):
        super().__init__(
            f"Using builtin type {dependency_type} as a top level dependency is not supported"
        )

