import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from inspect import Parameter

from ididi.errors import (
    GenericTypeNotSupportedError,
    MissingAnnotationError,
    UnsolvableDependencyError,
)
from ididi.utils.param_utils import NULL, Nullable, is_not_null
from ididi.utils.typing_utils import (
    eval_type,
    get_full_typed_signature,
    get_typed_annotation,
    is_builtin_type,
)


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
    depnode: "DependencyNode[ty.Any] | ForwardDependency", abstract_type: type
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
    raise NotImplementedError


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
            signature=None,
        )
    # Handle forward references
    elif isinstance(annotation, ty.ForwardRef):
        raise NotImplementedError("ForwardDependency should not be handled here")
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
            signature=get_full_typed_signature(annotation),
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
            node.dependencies.append(dependency)
        return node
    else:
        raise NotImplementedError(f"Unsupported annotation type: {annotation}")


class DependencyParam:
    """
    we have a 1-1-1-1 relationships
    node <-> dependent type <-> dependencies <-> signature
    DependencyNode[AuthService] <-> AuthService <-> dependencies <-> Signature.parameters
    """

    dependency: "DependencyNode[ty.Any]"
    param: inspect.Parameter


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyNode[T]:
    """
    A DAG node that represents a dependency

    dependent: the dependent type that this node represents, e.g AuthService -> AuthService
    factory: the factory function that creates the dependent, e.g cache_factory -> cache_factory, AuthService -> AuthService.__init__
    dependencies: the dependencies of the dependent, e.g cache_factory -> redis, keyspace
    """

    dependent: "ForwardDependent | type[T]"
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    dependencies: list["DependencyNode[ty.Any]"] = field(
        default_factory=list, repr=False
    )
    signature: inspect.Signature | None = field(repr=False)
    # we should combine dependencies and signature, as dependencies have a one to one relationship with signature parameters

    def _first_matched_impl(
        self, abstract_type: type
    ) -> "DependencyNode[ty.Any] | None":
        """
        Find the first dependency that matches the abstract type.
        """
        match_deps = (
            dep for dep in self.dependencies if is_implm_node(dep, abstract_type)
        )
        return next(match_deps, None)

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        """
        Build the dependent, resolving dependencies and applying defaults.
        kwargs override any dependencies or defaults.
        """
        if not self.dependencies:
            if is_not_null(self.default):
                return self.default
            elif isinstance(self.factory, type) and is_builtin_type(self.factory):
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)
            return self.factory(**kwargs)  # Apply overrides for no-dependency case

        signature = self.signature or get_full_typed_signature(self.factory)
        bound_args = signature.bind_partial()

        # First apply overrides
        for name, value in kwargs.items():
            if name in signature.parameters:
                bound_args.arguments[name] = value

        # Get parameters, skipping 'self' for classes
        params = signature.parameters
        if class_or_method(self.factory):
            params = dict(list(params.items())[1:])

        # Then resolve remaining dependencies
        for name, param in params.items():
            if name in bound_args.arguments:
                continue  # Skip if already bound or overridden

            required_type = param.annotation
            if is_builtin_type(required_type) and param.default == Parameter.empty:
                raise UnsolvableDependencyError(name, required_type)

            # Find matching dependency

            first_matched = self._first_matched_impl(required_type)

            # if self.dependent.__name__ == "Service2":
            #     breakpoint()

            if first_matched:
                # Don't pass overrides to dependencies
                built_dep = first_matched.build()
                bound_args.arguments[name] = built_dep
            elif param.default != Parameter.empty:
                bound_args.arguments[name] = param.default
            elif param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue  # Skip *args and **kwargs if not provided
            else:
                raise UnsolvableDependencyError(name, required_type)

        bound_args.apply_defaults()
        return self.factory(*bound_args.args, **bound_args.kwargs)

    @classmethod
    def _create_node[
        I
    ](
        cls,
        *,
        dependent: type[I],
        factory: ty.Callable[..., I],
        signature: inspect.Signature,
    ) -> "DependencyNode[I]":
        node = DependencyNode(dependent=dependent, factory=factory, signature=signature)
        params = tuple(signature.parameters.values())
        globalns = getattr(dependent.__init__, "__globals__", {})

        if class_or_method(factory):
            params = params[1:]

        for param in params:
            if isinstance(param.annotation, ty.ForwardRef):
                # if param is of type ForwardRef, we need to create a lazy evluated node
                lazy_dep = ForwardDependent(param.annotation, globalns)
                dep_node = ForwardDependency(
                    dependent=lazy_dep,
                    factory=lambda: None,
                    signature=None,
                )
                node.dependencies.append(dep_node)
            else:

                dependency = process_param(
                    dependent=dependent, param=param, globalns=globalns
                )
                node.dependencies.append(dependency)
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
        # remove if generic is supported
        raise_if_generic_type(dependent)

        if dependent.__init__ is object.__init__:
            return cls._create_node(
                dependent=dependent,
                factory=ty.cast(ty.Callable[..., I], dependent),
                signature=inspect.Signature(),  # Empty signature
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
            raise UnsolvableDependencyError(
                self.forward_ref.__forward_arg__, ty.ForwardRef
            ) from e


@dataclass(kw_only=True, slots=True, frozen=True)
class ForwardDependency(DependencyNode[ty.Any]):
    """
    ### Description:
    A special dependency node that lazily resolves forward references.

    ### Why:
    when we decorate a class that requires a forward reference with @dag.node,
    we would not be able to find the forward ref class in the __globals__ attribute of the __init__ method, before the forward ref class is not created yet.
    """

    dependent: ForwardDependent  # TODO: find a better way to type this

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
