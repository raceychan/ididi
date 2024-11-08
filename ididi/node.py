import abc
import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import Parameter

from .errors import (
    ABCNotImplementedError,
    CircularDependencyDetectedError,
    ForwardReferenceNotFoundError,
    GenericDependencyNotSupportedError,
    MissingAnnotationError,
    MissingReturnTypeError,
    NotSupportedError,
    ProtocolFacotryNotProvidedError,
    UnsolvableDependencyError,
)
from .types import EMPTY_SIGNATURE, INSPECT_EMPTY, IFactory, NodeConfig
from .utils.param_utils import NULL, Nullable, is_not_null
from .utils.typing_utils import (
    eval_type,
    get_factory_sig_from_cls,
    get_full_typed_signature,
    get_typed_annotation,
    get_typed_params,
    is_builtin_type,
    is_class,
    is_class_or_method,
    is_class_with_empty_init,
)


def factory_placeholder() -> None:
    raise NotSupportedError("Factory placeholder should not be called")


from abc import ABC, abstractmethod


@dataclass(slots=True)
class AbstractDependent[T](ABC):
    """Base class for all dependent types."""

    dependent_type: type[T] = field(init=False)

    @property
    @abstractmethod
    def dependent_name(self) -> str: ...


@dataclass(slots=True)
class Dependent[T](AbstractDependent[T]):
    """Represents a concrete (non-forward-reference) dependent type."""

    dependent_type: type[T]

    @property
    def dependent_name(self) -> str:
        return self.dependent_type.__name__


@dataclass(slots=True)
class ForwardDependent(AbstractDependent[ty.Any]):
    """A placeholder for a dependent type that hasn't been resolved yet."""

    forward_ref: ty.ForwardRef
    globalns: dict[str, ty.Any] = field(repr=False)

    @property
    def dependent_name(self) -> str:
        return self.forward_ref.__forward_arg__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.forward_ref})"

    def resolve(self) -> type:
        try:
            return eval_type(self.forward_ref, self.globalns, self.globalns)
        except NameError as e:
            raise ForwardReferenceNotFoundError(self.forward_ref) from e


@dataclass(slots=True)
class LazyDependent[T](AbstractDependent[T]):
    dependent_type: type[T]
    factory: IFactory[T, ...]
    signature: "DependentSignature[T]"
    cached_instance: Nullable[T] = NULL

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dependent_type})"

    def __getattr__(self, attrname: str, /) -> ty.Any:
        """
        dynamically build the dependent type on the fly
        """
        classattr = self.dependent_type.__dict__.get(attrname)

        try:
            if isinstance(classattr, property):
                raise KeyError(attrname)
            dpram = self.signature[attrname]
            return dpram.node.dependent
        except KeyError:
            if self.cached_instance is NULL:
                bound_args = self.signature.bound_args({})
                self.cached_instance = self.factory(
                    *bound_args.args, **bound_args.kwargs
                )
            return self.cached_instance.__getattribute__(attrname)

    @property
    def dependent_name(self) -> str:
        return self.dependent_type.__name__


# ======================= Signature =====================================


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyParam[T]:
    """'dpram' for short
    Represents a parameter and its corresponding dependency node

    @dg.node
    def dependent_factory(self, name: str) -> ty.Any:

    Here we 'name: str' would be built into a DependencyParam
    """

    name: str
    param: Parameter
    is_builtin: bool
    node: "DependentNode[T]"

    def __repr__(self) -> str:
        return f"{self.param.name}: {self.node}"

    @property
    def dependent_type(self) -> type[T]:
        return self.node.dependent_type

    @property
    def default(self) -> Nullable[T]:
        default_val = self.param.default
        if default_val is INSPECT_EMPTY:
            return NULL
        return default_val

    @property
    def should_not_resolve(self) -> bool:
        """
        Conditions that make this dependency param not be resolved at DependencyGraph.resolve()

        - lazy dependency
        Lazy Dependency should be resolved at runtime, not at graph resolution time
        - builtin type
        Builtin type can't be instantiated without arguments, so it should not be resolved
        """
        return self.node.config.lazy or self.is_builtin

    def build(self, override: T | dict[str, ty.Any]) -> T | ty.Awaitable[T]:
        """Build the dependency if needed, handling defaults and overrides."""
        if override:
            if override.__class__ is not dict:
                return ty.cast(T, override)
            elif self.dependent_type is dict:
                return ty.cast(T, override)

        if is_not_null(self.default):
            return self.default

        return self.node.build(**(ty.cast(dict[str, ty.Any], override)))


@dataclass(slots=True, frozen=True, kw_only=True)
class DependentSignature[T]:
    dependent: type[T | None] = field(default=type(None))
    dprams: dict[str, DependencyParam[ty.Any]] = field(default_factory=dict, repr=False)

    def __iter__(self) -> ty.Iterator[DependencyParam[ty.Any]]:
        return iter(self.dprams.values())

    def __len__(self) -> int:
        return len(self.dprams)

    def __getitem__(self, param_name: str) -> DependencyParam[ty.Any]:
        return self.dprams[param_name]

    def update(self, param_name: str, dpram: DependencyParam[ty.Any]) -> None:
        self.dprams[param_name] = dpram

    def bound_args(self, override: dict[str, ty.Any]) -> inspect.BoundArguments:
        dprams = self.dprams

        sig = inspect.Signature(
            parameters=[dpram.param for dpram in dprams.values()],
            return_annotation=self.dependent,
        )

        bound_args = sig.bind_partial()

        for dpram in dprams.values():
            if dpram.node.config.lazy:
                # TODO: do this in static resolve
                value = dpram.node.dependent
            else:
                param_override = override.pop(dpram.param.name, {})
                value = dpram.build(param_override)
            bound_args.arguments[dpram.param.name] = value

        bound_args.apply_defaults()
        return bound_args


# ======================= Node =====================================


def _create_holder_node[
    I
](
    dependent: AbstractDependent[I], default: Nullable[I], config: NodeConfig
) -> "DependentNode[ty.Any]":
    """
    Create a holder node for certain dependent types.
    e.g. builtin types, forward references, etc.
    """
    return DependentNode(
        dependent=dependent,
        factory=factory_placeholder,
        signature=DependentSignature(),
        default=default,
        config=config,
    )


def _create_sig[
    T
](
    *, dependent: type[T], params: ty.Sequence[Parameter], config: NodeConfig
) -> DependentSignature[T]:
    """
    Create a signature for a dependent type.
    """

    dep_params: dict[str, DependencyParam[ty.Any]] = {}
    globalns: dict[str, ty.Any] = getattr(dependent.__init__, "__globals__", {})

    for param in params:
        if is_builtin_type(dependent):
            dep = Dependent(dependent)
            dependency = _create_holder_node(dep, NULL, config)
            is_builtin = True
        elif isinstance(param.annotation, ty.ForwardRef):
            dependency = DependentNode.from_forwardref(
                forwardref=param.annotation, globalns=globalns, config=config
            )
            is_builtin = False
        else:
            dependency = DependentNode.from_param(
                dependent=dependent,
                param=param,
                globalns=globalns,
                config=config,
            )
            is_builtin = is_builtin_type(dependency.dependent.dependent_type)

        dep_param = DependencyParam(
            name=param.name,
            param=param,
            node=dependency,
            is_builtin=is_builtin,
        )
        dep_params[param.name] = dep_param

    return DependentSignature(dprams=dep_params, dependent=ty.cast(type[T], dependent))


@dataclass(kw_only=True, slots=True)
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
    - factory: the factory function that creates the dependent, e.g AuthService.__init__
    """

    dependent: AbstractDependent[T]
    default: Nullable[T] = NULL
    factory: ty.Callable[..., T]
    signature: DependentSignature[T]
    config: NodeConfig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dependent})"

    @property
    def dependent_type(self) -> type[T]:
        return self.dependent.dependent_type

    def resolve_forward_dependency(self, dep: ForwardDependent) -> type:
        """
        Resolve a forward dependency and check for circular dependencies.
        Returns the resolved type and its node.
        """

        resolved_type = dep.resolve()

        # Check for circular dependencies
        for sub_param in get_typed_params(resolved_type):
            if sub_param.annotation is self.dependent.dependent_type:
                raise CircularDependencyDetectedError(
                    [self.dependent.dependent_type, resolved_type]
                )

        return resolved_type

    def actualize_forward_dep(
        self, param: inspect.Parameter, dep: ForwardDependent
    ) -> "DependentNode[ty.Any]":
        resolved_type = self.resolve_forward_dependency(dep)
        resolved_node = DependentNode.from_node(resolved_type, self.config)
        new_dep_param = DependencyParam(
            name=param.name,
            param=param,
            node=resolved_node,
            is_builtin=False,
        )
        self.signature.update(param.name, new_dep_param)
        return resolved_node

    def iter_dependencies(self) -> ty.Generator["DependentNode[ty.Any]", None, None]:
        for dep_param in self.signature:
            if dep_param.is_builtin:
                continue

            if isinstance(dep_param.node.dependent, ForwardDependent):
                node = self.actualize_forward_dep(
                    dep_param.param, dep_param.node.dependent
                )
                yield node
            else:
                yield dep_param.node

    def check_for_resolvability(self) -> None:
        if isinstance(self.factory, type):
            # no factory override
            if is_builtin_type(self.factory):
                raise UnsolvableDependencyError(
                    self.dependent.dependent_name, self.factory
                )

            if getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)

            if issubclass(self.factory, abc.ABC) and (
                abstract_methods := getattr(self.factory, "__abstractmethods__", None)
            ):
                raise ABCNotImplementedError(self.factory, abstract_methods)

        for dpram in self.signature:
            if dpram.default:
                continue

            if dpram.is_builtin:
                raise UnsolvableDependencyError(
                    dpram.param.name, dpram.param.annotation
                )

            if dpram.param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                raise UnsolvableDependencyError(
                    dpram.param.name, dpram.param.annotation
                )

    def build(self, **override: dict[str, ty.Any]) -> T | ty.Awaitable[T]:
        """
        Build the dependent, resolving dependencies and applying defaults.
        kwargs override any dependencies or defaults.
        """
        self.check_for_resolvability()

        if not self.signature:
            return self.factory()

        bound_args = self.signature.bound_args(override)
        return self.factory(*bound_args.args, **bound_args.kwargs)

    @classmethod
    def create[
        **P, N
    ](
        cls,
        *,
        dependent: type[N],
        factory: ty.Callable[P, N],
        signature: inspect.Signature,
        config: NodeConfig,
    ) -> "DependentNode[N]":

        # TODO: we need to make errors happen within this function much more deatiled,
        # so that user can see right away what is wrong without having to trace back.

        params = tuple(signature.parameters.values())

        if is_class_or_method(factory):
            # Skip 'self' parameter
            params = params[1:]

        dpram_sig = _create_sig(dependent=dependent, params=params, config=config)

        if config.lazy:
            dep = LazyDependent(
                dependent_type=dependent, factory=factory, signature=dpram_sig
            )
        else:
            dep = Dependent(dependent_type=dependent)

        node = DependentNode(
            dependent=dep,
            factory=factory,
            signature=dpram_sig,
            config=config,
        )
        return node

    @classmethod
    def from_forwardref(
        cls, forwardref: ty.ForwardRef, globalns: dict[str, ty.Any], config: NodeConfig
    ) -> "DependentNode[ty.Any]":
        dep = ForwardDependent(forwardref, globalns)
        node = _create_holder_node(dependent=dep, default=NULL, config=config)
        return node

    @classmethod
    def from_param(
        cls,
        *,
        dependent: type[ty.Any],
        param: inspect.Parameter,
        globalns: dict[str, ty.Any],
        config: NodeConfig,
    ) -> "DependentNode[ty.Any]":
        """
        Process a parameter to create a dependency node.
        like def __init__(self, a: int, b: str):
        here we have two params: a and b

        we might use a fancier way to implement this, such as Chain of Responsibility,
        as we will have more and more cases to handle.
        Handles forward references, builtin types, classes, and generic types, etc.
        """

        param_name = param.name
        default = param.default if param.default != inspect.Parameter.empty else NULL
        annotation = get_typed_annotation(param.annotation, globalns)
        annotation = ty.get_origin(annotation) or annotation

        if annotation is inspect.Signature.empty:
            raise MissingAnnotationError(dependent, param_name)

        if isinstance(annotation, ty.TypeVar):
            # we would need to support generic types in the future
            # perhaps by storing its generic arguments in dependent node
            raise GenericDependencyNotSupportedError(annotation)

        # === Solvable dependency, NOTE: order matters!===

        if annotation is types.UnionType:
            union_types = ty.get_args(param.annotation)
            # we do care which type is correct, it would be provided by the factory
            annotation = union_types[0]

        if is_builtin_type(annotation):
            node = _create_holder_node(Dependent(annotation), default, config)
        else:
            node = DependentNode.from_node(annotation, config)
        return node

    @classmethod
    def from_factory[
        I, **P
    ](cls, *, factory: IFactory[I, P], config: NodeConfig) -> "DependentNode[I]":
        signature = get_full_typed_signature(factory)

        dependent: type[I] = signature.return_annotation
        if signature.return_annotation is inspect.Signature.empty:
            raise MissingReturnTypeError(factory)

        return cls.create(
            dependent=dependent, factory=factory, signature=signature, config=config
        )

    @classmethod
    def from_class[
        I
    ](cls, *, dependent: type[I], config: NodeConfig) -> "DependentNode[I]":
        if hasattr(dependent, "__origin__"):
            if res := ty.get_origin(dependent):
                dependent = res

        if is_class_with_empty_init(dependent):
            return cls.create(
                dependent=dependent,
                factory=ty.cast(ty.Callable[..., I], dependent),
                signature=EMPTY_SIGNATURE,
                config=config,
            )
        signature = get_factory_sig_from_cls(dependent)
        return cls.create(
            dependent=dependent, factory=dependent, signature=signature, config=config
        )

    @classmethod
    @lru_cache(maxsize=1000)
    def from_node[
        I, **P
    ](
        cls, node: IFactory[I, P] | type[I], config: NodeConfig | None = None
    ) -> "DependentNode[I]":
        config = config or NodeConfig()

        if is_class(node):
            return cls.from_class(dependent=node, config=config)
        else:
            if config.lazy:
                raise NotSupportedError(
                    "Lazy dependency is not supported for factories"
                )
            return cls.from_factory(
                factory=ty.cast(IFactory[I, P], node), config=config
            )
