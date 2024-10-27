import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from inspect import Parameter

from ididi.errors import (
    GenericTypeNotSupportedError,
    MissingAnnotationError,
    NotSupportedError,
    UnsolvableDependencyError,
)
from ididi.utils.param_utils import NULL, Nullable, is_not_null
from ididi.utils.typing_utils import (
    eval_type,
    get_full_typed_signature,
    get_typed_annotation,
    is_builtin_type,
)

EMPTY_SIGNATURE = inspect.Signature()


def class_or_method(obj: ty.Any) -> bool:
    return isinstance(obj, (type, types.MethodType))


def is_class(obj: type | ty.Callable[..., ty.Any]) -> bool:
    origin = ty.get_origin(obj) or obj
    is_type = isinstance(origin, type)
    is_generic_alias = isinstance(obj, types.GenericAlias)
    return is_type or is_generic_alias


def raise_if_generic_type(dependent: type) -> None:
    if hasattr(dependent, "__origin__"):
        raise GenericTypeNotSupportedError(dependent)


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
    default = NULL if param.default == inspect.Parameter.empty else param.default

    # Evaluate the annotation if it's a forward reference or string
    annotation = get_typed_annotation(param.annotation, globalns)

    if annotation is inspect.Parameter.empty:
        raise MissingAnnotationError(dependent, param_name)

    # Handle builtin types
    if is_builtin_type(annotation):
        if default is NULL:
            raise UnsolvableDependencyError(param_name, annotation)
        return DependencyNode(
            dependent=ty.cast(type[ty.Any], annotation),
            factory=annotation,
            default=default,
            # signature=None,
        )
    # Handle forward references
    elif isinstance(annotation, ty.ForwardRef):
        raise NotSupportedError("ForwardDependency should not be handled here")
    # Handle class types
    elif isinstance(annotation, type):
        return DependencyNode.from_node(annotation)
    # Handle generic types
    elif hasattr(annotation, "__origin__"):
        origin = ty.get_origin(annotation)
        args = ty.get_args(annotation)
        node = DependencyNode(
            dependent=origin,
            factory=annotation,
            default=default,
            # signature=get_full_typed_signature(annotation),
        )
        for i, arg in enumerate(args):
            dependency = process_param(
                dependent=dependent,
                param=inspect.Parameter(
                    f"{param_name}_arg{i}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=arg,
                ),
                globalns=globalns,
            )
            deparam = DependencyParam(
                name=param_name, param=param, dependency=dependency
            )
            node.dependency_params.append(deparam)
        return node
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


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyNode[T]:
    """
    A DAG node that represents a dependency

    dependent: the dependent type that this node represents, e.g AuthService -> AuthService
    factory: the factory function that creates the dependent, e.g cache_factory -> cache_factory, AuthService -> AuthService.__init__
    dependencies: the dependencies of the dependent, e.g cache_factory -> redis, keyspace
    """

    dependent: "type[T] | ForwardDependent"
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    dependency_params: list[DependencyParam[ty.Any]] = field(
        default_factory=list, repr=False
    )
    is_forward_dep: ty.ClassVar[bool] = False

    @property
    def signature(self) -> inspect.Signature:
        """Reconstruct signature from dependency_params"""
        if not self.dependency_params:
            return EMPTY_SIGNATURE
        parameters = [dep.param for dep in self.dependency_params]
        return inspect.Signature(
            parameters=parameters, return_annotation=self.dependent
        )

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        """
        Build the dependent, resolving dependencies and applying defaults.
        kwargs override any dependencies or defaults.
        """
        if not self.dependency_params:
            if is_not_null(self.default):
                return self.default
            elif isinstance(self.factory, type) and is_builtin_type(self.factory):
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)
            return self.factory()

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
        node = DependencyNode(
            dependent=dependent,
            factory=factory,
        )
        params = tuple(signature.parameters.values())
        globalns = getattr(dependent.__init__, "__globals__", {})

        if class_or_method(factory):
            params = params[1:]  # Skip 'self' parameter

        for param in params:
            if isinstance(param.annotation, ty.ForwardRef):
                # Handle forward references
                lazy_dep = ForwardDependent(param.annotation, globalns)
                dependency = ForwardDependency(
                    dependent=lazy_dep,
                    factory=lambda: None,
                )
                dep_param = DependencyParam(
                    name=param.name, param=param, dependency=dependency
                )
            else:
                # Handle regular dependencies
                dependency = process_param(
                    dependent=dependent,
                    param=param,
                    globalns=globalns,
                )
                dep_param = DependencyParam(
                    name=param.name, param=param, dependency=dependency
                )
            node.dependency_params.append(dep_param)

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
        # TODO: remove if generic is supported
        raise_if_generic_type(dependent)

        if dependent.__init__ is object.__init__:
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
            raise ValueError(f"Unsupported node type: {type(node)}")


@dataclass(frozen=True, slots=True)
class ForwardDependent:
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

    def resolve(self) -> type:
        """Resolve the forward reference to its actual type."""
        try:
            return eval_type(self.forward_ref, self.globalns, self.globalns)
        except NameError as e:
            # TODO: raise a better error message
            # You are trying to resolve a forward reference that has not been defined yet.
            raise UnsolvableDependencyError(
                self.forward_ref.__forward_arg__, ty.ForwardRef
            ) from e


@dataclass(kw_only=True, slots=True, frozen=True)
class ForwardDependency[T](DependencyNode[T]):
    """
    ### Description:
    A special dependency node that lazily resolves forward references.

    ### Why:
    when we decorate a class that requires a forward reference with @dag.node,
    we would not be able to find the forward ref class in the __globals__ attribute of the __init__ method, before the forward ref class is not created yet.
    """

    # TODO: find a better way than ForwardDependency to do this stuff, as it is not type safe
    dependent: ForwardDependent  # type: ignore
    is_forward_dep: ty.ClassVar[bool] = True

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        # Evaluate the forward reference at build time
        try:
            concrete_type = self.dependent.resolve()
            # Create a proper node now that we can resolve the type
            real_node = DependencyNode.from_node(concrete_type)
            return real_node.build(*args, **kwargs)
        except NameError as e:
            raise UnsolvableDependencyError(
                self.dependent.forward_ref.__forward_arg__, ty.ForwardRef
            ) from e
