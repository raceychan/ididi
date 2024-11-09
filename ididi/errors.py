import typing as ty
from inspect import Parameter


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


class _ErrorChain(ty.NamedTuple):
    prev: "_ErrorChain | None"
    error: NodeError


class NodeCreationError(NodeError):
    """
    Raised when a node can't be created.
    """

    def __init__(
        self,
        dependent: type | ty.Callable[..., ty.Any],
        *,
        error: Exception,
        param: Parameter | None = None,
        form_message: bool = False,
    ):
        self.dependent = dependent
        self.param = param
        self._root_cause = error

        # Build error chain using _ErrorNode
        if isinstance(error, NodeCreationError):
            self.error_chain = _ErrorChain(error.error_chain, self)
            self._root_cause = error._root_cause
        else:
            self.error_chain = _ErrorChain(None, self)

        super().__init__(self.form_error_chain() if form_message else "")

    @property
    def error(self) -> Exception:
        return self._root_cause

    def _make_error(self) -> str:
        if not self.param:
            return ""

        err = f"-> {self.dependent.__name__}({self.param.name}: {self.param.annotation.__class__.__name__})\n"
        return err

    def form_error_chain(self) -> str:
        """
        Forms an error chain showing the dependency path that led to the error.

        Example output:
        TokenBucketFactory(aiocache: RedisCache[str])
            -> RedisCache[str](redis: Redis)
                -> Redis(connection_pool: Optional[ConnectionPool])
                -> MissingAnnotationError: Unable to resolve dependency...
        """
        chain: list[str] = ["\n"]
        current_node = self.error_chain
        level = 0
        tab = " " * 4

        # Walk the linked list to build the chain
        while current_node:
            if isinstance(current_node.error, NodeCreationError):
                indent = tab * level
                chain.append(indent + current_node.error._make_error())
                level += 1
            current_node = current_node.prev

        # Add root cause at the end
        chain.append(
            tab * level + f"{self._root_cause.__class__.__name__}: {self._root_cause}"
        )

        return "".join(chain)


class UnsolvableParameterError(NodeError):
    """
    Raised when a parameter is unsolveable.
    """


class ForwardReferenceNotFoundError(UnsolvableParameterError):
    """
    Raised when a forward reference can't be found in the global namespace.
    """

    def __init__(self, forward_ref: ty.ForwardRef):
        msg = f"Unable to resolve forward reference: {forward_ref}, make sure it has been defined in the global namespace"
        super().__init__(msg)


class MissingAnnotationError(UnsolvableParameterError):
    def __init__(self, dependent: type, param_name: str):
        self.dependent = dependent
        self.param_name = param_name
        msg = f"Unable to resolve dependency for parameter: {param_name} in {dependent}, annotation for `{param_name}` must be provided"
        super().__init__(msg)


class MissingReturnTypeError(UnsolvableParameterError):
    """
    Raised when a factory has no return type.
    Thus can't be determined what it is trying to override
    """

    def __init__(self, factory: ty.Callable[..., ty.Any]):
        self.factory = factory
        msg = f"Factory {factory} must have a return type"
        super().__init__(msg)


class GenericDependencyNotSupportedError(NodeError):
    """
    Raised when attempting to use a generic type that is not yet supported.
    """

    def __init__(self, generic_type: type | ty.TypeVar):
        super().__init__(
            f"Using generic a type as a dependency is not yet supported: {generic_type}"
        )


class UnsolvableDependencyError(NodeError):
    """
    Raised when a dependency parameter can't be built.
    """

    def __init__(self, dep_name: str, required_type: ty.Any):
        self.message = f"Unable to resolve dependency for parameter: {dep_name}, value of {required_type} must be provided"
        super().__init__(self.message)


class ProtocolFacotryNotProvidedError(NodeError):
    """
    Raised when a protocol is used as a dependency without a factory.
    """

    def __init__(self, protocol: type):
        super().__init__(
            f"Protocol {protocol} can't be instantiated, a factory is required to resolve it"
        )


class ABCNotImplementedError(NodeError):
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
