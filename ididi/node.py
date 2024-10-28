import abc
import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from inspect import Parameter

from .errors import (
    ABCWithoutImplementationError,
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
    depnode: "DependencyNode[ty.Any] | ForwardDependency[ty.Any]", abstract_type: type
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
) -> "DependencyNode[ty.Any]":
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
        return DependencyNode(
            dependent=ty.cast(type[ty.Any], annotation),
            factory=annotation,
            default=default,
        )
    elif isinstance(annotation, type):
        return DependencyNode.from_node(annotation)
    else:
        raise NotSupportedError(f"Unsupported annotation type: {annotation}")


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyParam[T]:
    """Represents a parameter and its corresponding dependency node.

    This class encapsulates the relationship between a parameter and its
    dependency node, making the one-to-one relationship explicit.
    """

    name: str
    param: Parameter
    dependency: "DependencyNode[T] | ForwardDependency[T]"

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
    ) -> "DependencyParam[ty.Any]":
        if isinstance(param.annotation, ty.ForwardRef):
            lazy_dep = ty.cast(
                ForwardDependent[I], ForwardDependent(param.annotation, globalns)
            )
            dependency = ForwardDependency(
                dependent=lazy_dep,
                factory=lambda: None,
            )
            return DependencyParam(name=param.name, param=param, dependency=dependency)
        else:
            dependency = process_param(
                dependent=dependent,
                param=param,
                globalns=globalns,
            )
            return cls(name=param.name, param=param, dependency=dependency)


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyNode[T]:
    """
    A DAG node that represents a dependency

    dependent: the dependent type that this node represents, e.g AuthService -> AuthService
    factory: the factory function that creates the dependent, e.g cache_factory -> cache_factory, AuthService -> AuthService.__init__
    dependencies: the dependencies of the dependent, e.g cache_factory -> redis, keyspace
    """

    dependent: "type[T] | ForwardDependent[T]"
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    dependency_params: list[DependencyParam[ty.Any]] = field(
        default_factory=list, repr=False
    )

    @property
    def signature(self) -> inspect.Signature:
        """Reconstruct signature from dependency_params"""
        if not self.dependency_params:
            return EMPTY_SIGNATURE
        parameters = [dep.param for dep in self.dependency_params]
        return inspect.Signature(
            parameters=parameters, return_annotation=self.dependent
        )

    def build_type_without_init(self) -> T:
        """
        This is mostly for types that can't be directly constrctured,
        e.g abc.ABC, protocols, builtin types that can't be instantiated without arguments.
        """
        if is_not_null(self.default):
            return self.default
        elif isinstance(self.factory, type):
            if is_builtin_type(self.factory):
                # may be would can do something with container types, e.g list, dict, set, tuple?
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)
            elif getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)
            elif issubclass(self.factory, abc.ABC):
                if abstract_methods := self.factory.__abstractmethods__:
                    raise ABCWithoutImplementationError(self.factory, abstract_methods)
                return ty.cast(ty.Callable[..., T], self.factory)()
            return self.factory()
        else:
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
    ) -> "DependencyNode[N]":
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

        node = DependencyNode(
            dependent=dependent,
            factory=factory,
            dependency_params=dependency_params,
        )

        return node

    @classmethod
    def _from_factory[I](cls, factory: ty.Callable[..., I]) -> "DependencyNode[I]":
        signature = get_full_typed_signature(factory)
        if signature.return_annotation is inspect.Signature.empty:
            raise ValueError("Factory must have a return type")
        dependent = ty.cast(type[I], signature.return_annotation)
        return cls._create_node(
            dependent=dependent, factory=factory, signature=signature
        )

    @classmethod
    def _from_class[I](cls, dependent: type[I]) -> "DependencyNode[I]":
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
    def from_node[I](cls, node: type[I] | ty.Callable[..., I]) -> "DependencyNode[I]":
        if is_class(node):
            return cls._from_class(ty.cast(type[I], node))
        elif callable(node):
            return cls._from_factory(node)
        else:
            raise NotSupportedError(f"Unsupported node type: {type(node)}")


@dataclass(frozen=True, slots=True)
class ForwardDependent[T]:
    """
    A placeholder for a dependent type that hasn't been resolved yet.
    Holds the forward reference and the namespace needed to resolve it later.
    """

    forward_ref: ty.ForwardRef
    globalns: dict[str, ty.Any] = field(repr=False)

    @property
    def __name__(self) -> str:
        return self.forward_ref.__forward_arg__

    @property
    def __mro__(self) -> tuple[type, ...]:
        return (self.__class__, object)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ForwardDependent):
            return self.forward_ref.__forward_arg__ == other.forward_ref.__forward_arg__
        return False

    def __hash__(self) -> int:
        return hash(self.forward_ref.__forward_arg__)

    def resolve(self) -> type[T]:
        """Resolve the forward reference to its actual type."""
        try:
            return eval_type(self.forward_ref, self.globalns, self.globalns)
        except NameError as e:
            raise ForwardReferenceNotFoundError(self.forward_ref) from e


@dataclass(kw_only=True, slots=True, frozen=True)
class ForwardDependency[T](DependencyNode[T]):
    """
    ### Description:
    A special dependency node that lazily resolves forward references.

    ### Why:
    when we decorate a class that requires a forward reference with @dag.node,
    we would not be able to find the forward ref class in the __globals__ attribute of the __init__ method, before the forward ref class is defined.
    """

    dependent: ForwardDependent[T]

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        concrete_type = self.dependent.resolve()
        real_node = DependencyNode.from_node(concrete_type)
        return real_node.build(*args, **kwargs)
