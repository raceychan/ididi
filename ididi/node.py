import inspect
import typing as ty
from dataclasses import dataclass, field
from inspect import Parameter

from ididi.errors import UnsolvableDependencyError
from ididi.utils.param_utils import NULL, Nullable, is_not_null
from ididi.utils.typing_utils import get_full_typed_signature, is_builtin_type


def search_factory[
    I
](dependent: type[I], globalvars: dict[str, ty.Any] | None = None) -> (
    ty.Callable[..., I] | None
):
    # raise NotImplementedError
    globalns = getattr(dependent, "__globals__", globalvars) or {}
    for name, value in globalns.items():
        if name.startswith("_dt_"):  # and name.endswith("_factory"):
            sig = get_full_typed_signature(value)
            if sig.return_annotation == dependent:
                return value
    return None


def process_param(param: inspect.Parameter) -> "DependencyNode[ty.Any]":
    # shoud use search_factory to find the factory too
    annotation = param.annotation
    param_name = param.name

    # Handle default values
    default = NULL if param.default == inspect.Parameter.empty else param.default
    if is_builtin_type(annotation):
        if default is NULL:
            raise UnsolvableDependencyError(param_name, annotation)
        return DependencyNode(
            dependent=ty.cast(type[ty.Any], annotation),
            factory=annotation,
            default=default,
            signature=None,
        )
    elif inspect.isclass(annotation):
        return DependencyNode.from_node(annotation)

    elif hasattr(annotation, "__origin__"):  # For handling generic types
        # TODO: rewrite this, list[int] shoud just be list[int], not list, with dependency int
        # use inspect.isbuiltin to check if the origin is a builtin type
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
                inspect.Parameter(
                    f"{param_name}_arg{i}",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=arg,
                )
            )
            node.dependencies.append(dependency)
        return node
    else:
        raise ValueError(f"Unsupported annotation type: {annotation}")


@dataclass(kw_only=True)
class DependencyNode[T]:
    """
    A DAG node that represents a dependency

    name: name of the dependent, e.g cache_factory -> "cache_factory"
    dependent: the callable that this node represents, e.g cache_factory -> cache_factory, AuthService -> AuthService
    dependencies: the dependencies of the dependent, e.g cache_factory -> redis, keyspace
    """

    dependent: type[T]
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    dependencies: list["DependencyNode[ty.Any]"] = field(default_factory=list)
    signature: inspect.Signature | None  # builtin types do not have signature

    def build(self, *args, **kwargs) -> T:
        """
        Build the dependent, resolving dependencies and applying defaults
        """
        if not self.dependencies:
            if is_not_null(self.default):
                return self.default
            elif isinstance(self.factory, type) and is_builtin_type(self.factory):
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)
            return self.factory()

        signature = self.signature or get_full_typed_signature(self.factory)
        bound_args = signature.bind_partial()

        # Get parameters, skipping 'self' for classes
        params = signature.parameters
        if inspect.isclass(self.factory) or inspect.ismethod(self.factory):
            params = dict(list(params.items())[1:])

        for name, param in params.items():
            if name in bound_args.arguments:
                continue  # Skip if already bound

            required_type = param.annotation
            if is_builtin_type(required_type) and param.default == Parameter.empty:
                raise UnsolvableDependencyError(name, required_type)

            matching_dep = next(
                (
                    dep
                    for dep in self.dependencies
                    if issubclass(dep.dependent, required_type)
                ),
                None,
            )

            if matching_dep:
                built_dep = matching_dep.build()
                bound_args.arguments[name] = built_dep
            elif param.default != Parameter.empty:
                bound_args.arguments[name] = param.default
            elif param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue  # Skip *args and **kwargs if not provided
            else:
                raise UnsolvableDependencyError(name, required_type)

        bound_args.apply_defaults()
        try:
            return self.factory(*bound_args.args, **bound_args.kwargs)
        except Exception:
            raise

    @classmethod
    def _create_node[
        I
    ](
        cls,
        dependent: type[I],
        factory: ty.Callable[..., I],
        signature: inspect.Signature,
    ) -> "DependencyNode[I]":
        node = DependencyNode(dependent=dependent, factory=factory, signature=signature)
        params = tuple(signature.parameters.values())

        if inspect.isclass(factory) or inspect.ismethod(factory):
            params = params[1:]

        for param in params:
            node.dependencies.append(process_param(param))
        return node

    @classmethod
    def _from_factory[I](cls, factory: ty.Callable[..., I]) -> "DependencyNode[I]":
        signature = get_full_typed_signature(factory)
        if signature.return_annotation is inspect.Signature.empty:
            raise ValueError("Factory must have a return type")
        dependent = ty.cast(type[I], signature.return_annotation)
        return cls._create_node(dependent, factory, signature)

    @classmethod
    def _from_class[I](cls, dependent: type[I]) -> "DependencyNode[I]":
        factory = search_factory(dependent)
        if factory is None:
            signature = get_full_typed_signature(dependent.__init__)
            signature = signature.replace(return_annotation=dependent)
            factory = ty.cast(ty.Callable[..., I], dependent)
        else:
            signature = get_full_typed_signature(factory)
            if signature.return_annotation is inspect.Signature.empty:
                raise ValueError("Factory must have a return type")
        return cls._create_node(dependent, factory, signature)

    @classmethod
    def from_node[I](cls, node: type[I] | ty.Callable[..., I]) -> "DependencyNode[I]":
        if callable(node) and not inspect.isclass(node):
            return cls._from_factory(node)
        return cls._from_class(ty.cast(type[I], node))
