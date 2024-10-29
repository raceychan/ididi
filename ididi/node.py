import abc
import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from inspect import Parameter

from .errors import (
    ABCWithoutImplementationError,
    CircularDependencyDetectedError,
    ForwardReferenceNotFoundError,
    GenericDependencyNotSupportedError,
    MissingAnnotationError,
    NotSupportedError,
    ProtocolFacotryNotProvidedError,
    UnsolvableDependencyError,
)
from .utils.param_utils import NULL, Nullable, is_not_null
from .utils.typing_utils import (
    eval_type,
    get_full_typed_signature,
    get_typed_annotation,
    is_builtin_type,
)

EMPTY_SIGNATURE = inspect.Signature()


def is_class_or_method(obj: ty.Any) -> bool:
    return isinstance(obj, (type, types.MethodType))


def is_class(obj: type | ty.Callable[..., ty.Any]) -> bool:
    origin = ty.get_origin(obj) or obj
    is_type = isinstance(origin, type)
    is_generic_alias = isinstance(obj, types.GenericAlias)
    return is_type or is_generic_alias


def is_class_with_empty_init(cls: type) -> bool:
    is_undefined_init = cls.__init__ is object.__init__
    is_protocol = type(cls) is ty._ProtocolMeta  # type: ignore
    return is_undefined_init or is_protocol


def is_implm_node(
    depnode: "DependentNode[ty.Any] | ForwardDependentNode[ty.Any]", abstract_type: type
) -> bool:
    """
    Check if the dependent type of the dependency node is a valid implementation of the abstract type.
    """
    dep_type = depnode.dependent
    return isinstance(dep_type, ForwardDependent) or issubclass(dep_type, abstract_type)


def search_factory[
    I
](dependent: type[I], globalvars: dict[str, ty.Any] | None = None) -> (
    ty.Callable[..., I] | None
):
    raise NotSupportedError("searching to be implemented")


def process_param[
    I
](
    *,
    dependent: type[I],
    param: inspect.Parameter,
    globalns: dict[str, ty.Any],
) -> "DependentNode[ty.Any]":
    """
    Process a parameter to create a dependency node.
    Handles forward references, builtin types, classes, and generic types.
    """
    param_name = param.name
    default = param.default if param.default != inspect.Parameter.empty else NULL

    annotation = get_typed_annotation(param.annotation, globalns)

    if hasattr(annotation, "__origin__"):
        annotation = ty.get_origin(annotation)

    if annotation is inspect.Parameter.empty:
        raise MissingAnnotationError(dependent, param_name)

    if isinstance(annotation, ty.ForwardRef):
        raise NotSupportedError("ForwardDependency should not be handled here")

    if isinstance(annotation, ty.TypeVar):
        raise GenericDependencyNotSupportedError(annotation)

    # Handle builtin types
    if is_builtin_type(annotation):
        return DependentNode(
            dependent=ty.cast(type[ty.Any], annotation),
            factory=annotation,
            default=default,
        )
    elif isinstance(annotation, type):
        return DependentNode.from_node(annotation)
    else:
        raise NotSupportedError(f"Unsupported annotation type: {annotation}")


@dataclass(frozen=True, slots=True)
class AbstractDependent[T]:
    """Base class for all dependent types."""

    @property
    def __name__(self) -> str:
        """Return the name of the dependent type."""
        raise NotImplementedError

    @property
    def __mro__(self) -> tuple[type, ...]:
        """Return the method resolution order."""
        raise NotImplementedError

    def resolve(self) -> type[T]:
        """Resolve to concrete type."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Dependent[T](AbstractDependent[T]):
    """Represents a concrete (non-forward-reference) dependent type."""

    type_: type[T]

    @property
    def __name__(self) -> str:
        return self.type_.__name__

    def resolve(self) -> type[T]:
        return self.type_

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dependent):
            return False
        return self.type_ == ty.cast(Dependent[ty.Any], other).type_

    def __hash__(self) -> int:
        return hash(self.type_)


@dataclass(frozen=True, slots=True)
class ForwardDependent(AbstractDependent[ty.Any]):
    """A placeholder for a dependent type that hasn't been resolved yet."""

    forward_ref: ty.ForwardRef
    globalns: dict[str, ty.Any] = field(repr=False)

    @property
    def __name__(self) -> str:
        return self.forward_ref.__forward_arg__

    @property
    def __mro__(self) -> tuple[type, ...]:
        return (self.__class__, object)

    def resolve(self) -> type:
        try:
            return eval_type(self.forward_ref, self.globalns, self.globalns)
        except NameError as e:
            raise ForwardReferenceNotFoundError(self.forward_ref) from e

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ForwardDependent):
            return self.forward_ref.__forward_arg__ == other.forward_ref.__forward_arg__
        return False

    def __hash__(self) -> int:
        return hash(self.forward_ref.__forward_arg__)


"""
class LazyDependent(AbstractDependent[ty.Any]):
    def __getattr__(self, name: str) -> ty.Any:
        '''
        dynamically build the dependent type on the fly
        '''

    def resolve(self) -> ty.Self:
        return self
"""


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyParam:
    """Represents a parameter and its corresponding dependency node.

    This class encapsulates the relationship between a parameter and its
    dependency node, making the one-to-one relationship explicit.
    """

    name: str
    param: Parameter
    dependency: "DependentNode[ty.Any]"

    def build(self, **kwargs: ty.Any) -> ty.Any:
        """Build the dependency if needed, handling defaults and overrides."""
        if self.param.name in kwargs:
            return kwargs[self.param.name]

        if self.param.default != Parameter.empty:
            return self.param.default

        if self.param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            return {}  # Skip *args and **kwargs if not provided

        return self.dependency.build()

    @classmethod
    def from_param[
        I
    ](
        cls, dependent: type[I], param: inspect.Parameter, globalns: dict[str, ty.Any]
    ) -> "DependencyParam":
        if isinstance(param.annotation, ty.ForwardRef):
            lazy_dep = ForwardDependent(param.annotation, globalns)
            dependency = ForwardDependentNode(
                dependent=lazy_dep,
                factory=lambda: None,
            )
            return cls(name=param.name, param=param, dependency=dependency)
        else:
            dependency = process_param(
                dependent=dependent,
                param=param,
                globalns=globalns,
            )
            return cls(name=param.name, param=param, dependency=dependency)


@dataclass(kw_only=True, slots=True, frozen=True)
class DependentNode[T]:
    """
    A DAG node that represents a dependency

    Examples:
    -----
    ```python
    @dag.node
    class AuthService:
        def __init__(self, redis: Redis, keyspace: str):
            pass
    ```

    in this case:
    - dependent: the dependent type that this node represents, e.g AuthService
    - factory: the factory function that creates the dependent, e.g AuthService.__init__, auth_service_factory
    - dependencies: the dependencies of the dependent, e.g redis: Redis, keyspace: str
    """

    dependent: "type[T] | ForwardDependent"
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    dependency_params: list[DependencyParam] = field(default_factory=list, repr=False)

    @property
    def signature(self) -> inspect.Signature:
        """Reconstruct signature from dependency_params"""
        if not self.dependency_params:
            return EMPTY_SIGNATURE
        parameters = [dep.param for dep in self.dependency_params]
        return inspect.Signature(
            parameters=parameters, return_annotation=self.dependent
        )

    def resolve_forward_dependency(
        self, dep: ForwardDependent, concrete_type: type
    ) -> type:
        """
        Resolve a forward dependency and check for circular dependencies.
        Returns the resolved type and its node.
        """
        resolved_type = dep.resolve()

        # Check for circular dependencies
        sub_params = get_full_typed_signature(resolved_type).parameters.values()
        for sub_param in sub_params:
            if sub_param.annotation is concrete_type:
                raise CircularDependencyDetectedError([concrete_type, resolved_type])

        return resolved_type

    def get_dependency_resolution_info(
        self, concrete_type: type
    ) -> list[tuple[Parameter, "DependentNode[ty.Any]"]]:
        """
        Get a list of parameters and their resolved dependency types.
        """
        resolution_info: list[tuple[Parameter, "DependentNode[ty.Any]"]] = []

        for dep_param in self.dependency_params:
            param = dep_param.param
            dep = dep_param.dependency.dependent

            if is_builtin_type(dep):
                continue

            if isinstance(dep, ForwardDependent):
                resolved_type = self.resolve_forward_dependency(dep, concrete_type)
            else:
                resolved_type = dep

            resolved_node = DependentNode.from_node(resolved_type)
            resolution_info.append((param, resolved_node))
        return resolution_info

    def build_type_without_init(self) -> T:
        """
        This is mostly for types that can't be directly constrctured,
        e.g abc.ABC, protocols, builtin types that can't be instantiated without arguments.
        """
        if is_not_null(self.default):
            return self.default

        if isinstance(self.factory, type):
            if is_builtin_type(self.factory):
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)
            elif getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)
            elif issubclass(self.factory, abc.ABC):
                if abstract_methods := self.factory.__abstractmethods__:
                    raise ABCWithoutImplementationError(self.factory, abstract_methods)
                return ty.cast(ty.Callable[..., T], self.factory)()
            else:
                return ty.cast(type[T], self.factory)()

        try:
            return self.factory()
        except Exception as e:
            raise UnsolvableDependencyError(
                str(self.dependent), type(self.factory)
            ) from e

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        """
        Build the dependent, resolving dependencies and applying defaults.
        kwargs override any dependencies or defaults.
        """
        if not self.dependency_params:
            return self.build_type_without_init()

        bound_args = self.signature.bind_partial()

        # Build all dependencies using DependencyParam
        for dep_param in self.dependency_params:
            value = dep_param.build(*args, **kwargs)
            if value is not None:  # Skip None values from *args/**kwargs
                bound_args.arguments[dep_param.param.name] = value

        bound_args.apply_defaults()
        return self.factory(*bound_args.args, **bound_args.kwargs)

    @classmethod
    def _create_node[
        N
    ](
        cls,
        *,
        dependent: type[N],
        factory: ty.Callable[..., N],
        signature: inspect.Signature,
    ) -> "DependentNode[N]":
        """Create a new dependency node with the specified type."""

        # TODO: we need to make errors happen within this function much more deatiled,
        # so that user can see right away what is wrong without having to trace back.

        params = tuple(signature.parameters.values())
        globalns = getattr(dependent.__init__, "__globals__", {})

        if is_class_or_method(factory):
            # Skip 'self' parameter
            params = params[1:]

        dependency_params = list(
            DependencyParam.from_param(dependent, param, globalns) for param in params
        )

        node = DependentNode(
            dependent=dependent,
            factory=factory,
            dependency_params=dependency_params,
        )

        return node

    @classmethod
    def _from_factory[I](cls, factory: ty.Callable[..., I]) -> "DependentNode[I]":
        signature = get_full_typed_signature(factory)
        if signature.return_annotation is inspect.Signature.empty:
            raise ValueError("Factory must have a return type")
        dependent = ty.cast(type[I], signature.return_annotation)
        return cls._create_node(
            dependent=dependent, factory=factory, signature=signature
        )

    @classmethod
    def _from_class[I](cls, dependent: type[I]) -> "DependentNode[I]":
        if hasattr(dependent, "__origin__"):
            if res := ty.get_origin(dependent):
                dependent = res

        if is_class_with_empty_init(dependent):
            # a class without __init__
            return cls._create_node(
                dependent=dependent,
                factory=ty.cast(ty.Callable[..., I], dependent),
                signature=EMPTY_SIGNATURE,
            )
        signature = get_full_typed_signature(dependent.__init__)
        signature = signature.replace(return_annotation=dependent)
        return cls._create_node(
            dependent=dependent, factory=dependent, signature=signature
        )

    @classmethod
    def from_node[I](cls, node: type[I] | ty.Callable[..., I]) -> "DependentNode[I]":
        if is_class(node):
            return cls._from_class(ty.cast(type[I], node))
        elif callable(node):
            return cls._from_factory(node)
        else:
            raise NotSupportedError(f"Unsupported node type: {type(node)}")


@dataclass(kw_only=True, slots=True, frozen=True)
class ForwardDependentNode[T](DependentNode[T]):
    """
    ### Description:
    A special dependency node that lazily resolves forward references.

    ### Why:
    when we decorate a class that requires a forward reference with @dag.node,
    we would not be able to find the forward ref class in the __globals__ attribute of the __init__ method, before the forward ref class is defined.
    """

    dependent: ForwardDependent

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        concrete_type = self.dependent.resolve()
        real_node = DependentNode.from_node(concrete_type)
        return real_node.build(*args, **kwargs)


"""

class LazyDependentNode(AbstractDependent[ty.Any]):
    dependent: LazyDependent
"""
