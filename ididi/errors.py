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


# TODO: 1. differentaite node creation and node resolve errors
# TODO: 2. make NodeErrorChain more generic


class _ErrorChain:
    __slots__ = ("error", "next", "head", "length")

    error: Exception
    next: "_ErrorChain | None"
    head: "_ErrorChain"
    length: int

    def __init__(self, *, error: Exception):
        self.error = error
        if isinstance(error, NodeCreationErrorChain):
            self.next = error.error_chain
            self.length = error.error_chain.length + 1
            self.head = error.error_chain.head
        else:
            self.next = None
            self.length = 0
            self.head = self

    def __str__(self) -> str:
        return self.form_repr()

    def restart(self) -> None:
        """
        Restarts the error chain from the head.
        """

        head = self.head
        self.next = head.next
        self.head = self
        del head

    def _make_error(self) -> str:
        """Format the current error node's message."""

        # TODO: 1. skip middle traceback
        # TODO: 2. _make_error should be an method of NodeCreationErrorChain
        if not isinstance(self.error, NodeCreationErrorChain):
            return ""

        if not self.error.param:
            return ""

        param_type = self.error.param.annotation
        if param_type is Parameter.empty:
            param_type = "ididi.Missing"
        else:
            try:
                param_type = param_type.__name__
            except AttributeError:
                param_type = str(param_type)

        return f"-> {self.error.dependent.__name__}({self.error.param.name}: {param_type})\n"

    def form_repr(self) -> str:
        """
        Forms an error chain showing the dependency path that led to the error.

        Example output:
        TokenBucketFactory(aiocache: RedisCache[str])
        -> RedisCache[str](redis: Redis)
        -> Redis(connection_pool: Optional[ConnectionPool])
        -> MissingAnnotationError: Unable to resolve dependency...
        """
        messages: list[str] = ["\n"]
        current_node = self
        level = 0

        # Walk the linked list to build the chain
        while current_node:
            msg = current_node._make_error()
            if msg:
                messages.append(msg)
                level += 1
            current_node = current_node.next

        # Add root cause at the end
        messages.append(f"{self.head.error.__class__.__name__}: {self.head.error}")
        return "".join(messages)


class AsyncResourceInSyncError(NodeError):
    def __init__(self, factory: ty.Callable[..., ty.Any]):
        self.factory = factory
        super().__init__(
            f"Requiring async resource {factory} in a sync context is not supported"
        )


class NodeCreationErrorChain(NodeError):
    """
    Raised when a node can't be created.
    This chain up the errors happened in a deep recursion stack.
    the root cause can be found in the .error attribute.
    """

    __slots__ = ("dependent", "param", "_root_cause", "error_chain")

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
        self.error_chain = _ErrorChain(error=error)
        self._root_cause = self.error_chain.head.error
        message = ""
        if form_message:
            message = self.error_chain.form_repr()

        super().__init__(message)

    @property
    def error(self) -> Exception:
        return self._root_cause

    def with_error_chain(self) -> "NodeCreationErrorChain":
        """Returns a new instance with the error chain formatted in the message."""
        return NodeCreationErrorChain(
            dependent=self.dependent,
            error=self.error,
            param=self.param,
            form_message=True,
        )


class NodeResolveError(IDIDIError):
    """
    Base class for all node resolve related exceptions.
    """


class PositionalOverrideError(NodeResolveError):
    """
    Raised when a positional override is used.
    """

    def __init__(self, args: tuple[ty.Any, ...]):
        super().__init__(
            f"Positional overrides {args} are not supported, use keyword arguments instead"
        )


class UnsolvableNode(NodeResolveError): ...


class UnsolvableDependencyError(UnsolvableNode):
    """
    Raised when a dependency parameter can't be built.
    """

    def __init__(
        self,
        *,
        dep_name: str,
        factory: ty.Callable[..., ty.Any] | type,
        required_type: type,
    ):
        self.message = f"Unable to resolve dependency for parameter: {dep_name} in {factory}, value of {required_type} must be provided"
        super().__init__(self.message)


class ProtocolFacotryNotProvidedError(UnsolvableNode):
    """
    Raised when a protocol is used as a dependency without a factory.
    """

    def __init__(self, protocol: type):
        super().__init__(
            f"Protocol {protocol} can't be instantiated, a factory is required to resolve it"
        )


class ABCNotImplementedError(UnsolvableNode):
    """
    Raised when an ABC is used as a dependency without a factory.
    """

    def __init__(self, abc: type, abstract_methods: frozenset[str]):
        super().__init__(
            f"ABC {abc} has no valid implementations, either provide a implementation that implements {abstract_methods} or a factory"
        )


class UnsolvableParameterError(UnsolvableNode):
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


# =============== Graph Errors ===============
class GraphError(IDIDIError):
    """
    Base class for all graph related exceptions.
    """


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


class TopLevelBulitinTypeError(GraphResolveError):
    """
    Raised when a builtin type is used as a top level dependency.
    Example:
    >>> dag.resolve(int)
    """

    def __init__(self, dependency_type: type):
        super().__init__(
            f"Using builtin type {dependency_type} as a top level dependency is not supported"
        )


class MissingImplementationError(GraphResolveError):
    """
    Raised when a type has no implementations.
    """

    def __init__(self, dependency_type: type):
        super().__init__(f"No implementations found for {dependency_type}")
