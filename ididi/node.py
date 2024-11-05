import abc
import inspect
import types
import typing as ty
from dataclasses import dataclass, field
from functools import lru_cache
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
from .types import IFactory, NodeConfig
from .utils.param_utils import NULL, Nullable
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


def factory_placeholder() -> None:
    raise NotSupportedError("Factory placeholder should not be called")


@dataclass(slots=True)
class AbstractDependent[T]:
    """Base class for all dependent types."""

    dependent_type: type[T] = field(init=False)

    @property
    def __name__(self) -> str:
        return "AbstractDependent"


@dataclass(slots=True)
class Dependent[T](AbstractDependent[T]):
    """Represents a concrete (non-forward-reference) dependent type."""

    dependent_type: type[T]

    @property
    def __name__(self) -> str:
        return self.dependent_type.__name__


@dataclass(slots=True)
class ForwardDependent(AbstractDependent[ty.Any]):
    """A placeholder for a dependent type that hasn't been resolved yet."""

    forward_ref: ty.ForwardRef
    globalns: dict[str, ty.Any] = field(repr=False)

    def resolve(self) -> type:
        try:
            return eval_type(self.forward_ref, self.globalns, self.globalns)
        except NameError as e:
            raise ForwardReferenceNotFoundError(self.forward_ref) from e


@dataclass(slots=True)
class LazyDependent[T](AbstractDependent[T]):
    dependent_type: type[T]
    signature: "DependentSignature"
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
                bound_args = self.signature.bound_args()
                self.cached_instance = self.dependent_type(
                    *bound_args.args, **bound_args.kwargs
                )
            return self.cached_instance.__getattribute__(attrname)


# ======================= Signature =====================================


@dataclass(kw_only=True, slots=True, frozen=True)
class DependencyParam:
    """'dpram' for short
    Represents a parameter and its corresponding dependency node

    @dg.node
    def dependent_factory(self, name: str) -> ty.Any:

    Here we 'name: str' would be built into a DependencyParam
    """

    name: str
    param: Parameter
    is_builtin: bool
    node: "DependentNode[ty.Any]"

    def __repr__(self) -> str:
        return f"{self.param.name}: {self.node}"

    @property
    def dependent_type(self) -> type[ty.Any]:
        return self.node.dependent_type

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

    def build(self, **kwargs: ty.Any) -> ty.Any:
        """Build the dependency if needed, handling defaults and overrides."""

        if self.param.name in kwargs:
            return kwargs[self.param.name]

        if self.param.default != Parameter.empty:
            return self.param.default

        if self.is_builtin:
            raise UnsolvableDependencyError(self.param.name, self.param.annotation)

        if self.param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            raise UnsolvableDependencyError(self.param.name, self.param.annotation)

        return self.node.build()


@dataclass(slots=True, frozen=True, kw_only=True)
class DependentSignature:
    dprams: dict[str, DependencyParam] = field(default_factory=dict, repr=False)
    dependent: type[ty.Any] = type(None)

    def __iter__(self) -> ty.Iterator[DependencyParam]:
        return iter(self.dprams.values())

    def __len__(self) -> int:
        return len(self.dprams)

    def __getitem__(self, param_name: str) -> DependencyParam:
        return self.dprams[param_name]

    def update(self, param_name: str, dpram: DependencyParam) -> None:
        self.dprams[param_name] = dpram

    def bound_args(self, *args: ty.Any, **kwargs: ty.Any) -> inspect.BoundArguments:
        dprams = self.dprams

        sig = inspect.Signature(
            parameters=[dpram.param for dpram in dprams.values()],
            return_annotation=self.dependent,
        )

        bound_args = sig.bind_partial()

        for dpram in dprams.values():
            # TODO: make this in static resolve
            if dpram.node.config.lazy:
                value = dpram.node.dependent
            else:
                value = dpram.build(*args, **kwargs)
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
    if default is inspect.Parameter.empty:
        default = NULL
    return DependentNode(
        dependent=dependent,
        factory=factory_placeholder,
        signature=DependentSignature(),
        default=default,
        config=config,
    )


def _create_sig(
    *, dependent: type[ty.Any], params: ty.Sequence[Parameter], config: NodeConfig
) -> DependentSignature:
    """
    Create a signature for a dependent type.
    """

    dep_params: dict[str, DependencyParam] = {}
    globalns: dict[str, ty.Any] = getattr(dependent.__init__, "__globals__", {})

    for param in params:
        if is_builtin_type(dependent):
            dep = Dependent(dependent)
            dependency = _create_holder_node(dep, NULL, config)
            is_builtin = True
        elif isinstance(param.annotation, ty.ForwardRef):
            dep = ForwardDependent(param.annotation, globalns)
            dependency = _create_holder_node(
                dependent=dep, default=param.default, config=config
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
    return DependentSignature(dprams=dep_params, dependent=dependent)


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
    signature: DependentSignature
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
        sub_params = get_full_typed_signature(resolved_type).parameters.values()
        for sub_param in sub_params:
            if sub_param.annotation is self.dependent.dependent_type:
                raise CircularDependencyDetectedError(
                    [self.dependent.dependent_type, resolved_type]
                )

        return resolved_type

    def actualize_forward_deps(
        self,
    ) -> ty.Generator["DependentNode[ty.Any]", None, None]:
        """
        Update all forward dependency params to their resolved types.
        """

        for dep_param in self.signature:
            param = dep_param.param
            dep = dep_param.node.dependent

            if dep_param.is_builtin:
                continue

            if not isinstance(dep, ForwardDependent):
                continue

            resolved_type = self.resolve_forward_dependency(dep)
            resolved_node = DependentNode.from_node(resolved_type, self.config)
            new_dep_param = DependencyParam(
                name=param.name,
                param=param,
                node=resolved_node,
                is_builtin=False,
            )
            self.signature.update(param.name, new_dep_param)
            yield resolved_node

    def iter_dependencies(self) -> ty.Generator["DependentNode[ty.Any]", None, None]:
        for dep_param in self.signature:
            if dep_param.is_builtin:
                continue
            yield dep_param.node

    def build_type_without_dependencies(self) -> T:
        """
        This is mostly for types that can't be directly constrctured,
        e.g abc.ABC, protocols, builtin types that can't be instantiated without arguments.
        """
        if isinstance(self.factory, type):
            if is_builtin_type(self.factory):
                raise UnsolvableDependencyError(self.dependent.__name__, self.factory)

            if getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)

            if issubclass(self.factory, abc.ABC):
                if abstract_methods := getattr(
                    self.factory, "__abstractmethods__", None
                ):
                    raise ABCWithoutImplementationError(self.factory, abstract_methods)
                return ty.cast(ty.Callable[..., T], self.factory)()

        return self.factory()

    def build(self, *args: ty.Any, **kwargs: ty.Any) -> T:
        """
        Build the dependent, resolving dependencies and applying defaults.
        kwargs override any dependencies or defaults.
        """
        if not self.signature:
            return self.build_type_without_dependencies()

        bound_args = self.signature.bound_args(*args, **kwargs)
        return self.factory(*bound_args.args, **bound_args.kwargs)

    @classmethod
    def _create[
        N
    ](
        cls,
        *,
        dependent: type[N],
        factory: ty.Callable[..., N],
        signature: inspect.Signature,
        config: NodeConfig,
    ) -> "DependentNode[N]":
        """Create a new dependency node with the specified type."""

        # TODO: we need to make errors happen within this function much more deatiled,
        # so that user can see right away what is wrong without having to trace back.

        params = tuple(signature.parameters.values())

        if is_class_or_method(factory):
            # Skip 'self' parameter
            params = params[1:]

        dpram_sig = _create_sig(dependent=dependent, params=params, config=config)

        if config.lazy:
            dep = LazyDependent(dependent_type=dependent, signature=dpram_sig)
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

        if annotation is inspect.Parameter.empty:
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.KEYWORD_ONLY):
                raise NotSupportedError(
                    f"Unsupported parameter kind: {param.kind} in {dependent}"
                )
            raise MissingAnnotationError(dependent, param_name)

        if isinstance(annotation, ty.TypeVar):
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
    def _from_factory[
        I, **P
    ](cls, factory: IFactory[I, P], config: NodeConfig) -> "DependentNode[I]":

        signature = get_full_typed_signature(factory)
        if signature.return_annotation is inspect.Signature.empty:
            raise ValueError("Factory must have a return type")
        dependent = ty.cast(type[I], signature.return_annotation)

        return cls._create(
            dependent=dependent, factory=factory, signature=signature, config=config
        )

    @classmethod
    def _from_class[
        I
    ](cls, dependent: type[I], config: NodeConfig) -> "DependentNode[I]":
        if hasattr(dependent, "__origin__"):
            if res := ty.get_origin(dependent):
                dependent = res

        if is_class_with_empty_init(dependent):
            return cls._create(
                dependent=dependent,
                factory=ty.cast(ty.Callable[..., I], dependent),
                signature=EMPTY_SIGNATURE,
                config=config,
            )
        signature = get_full_typed_signature(dependent.__init__)
        signature = signature.replace(return_annotation=dependent)
        return cls._create(
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
            return cls._from_class(ty.cast(type[I], node), config)
        elif callable(node):
            if config.lazy:
                raise NotSupportedError(
                    "Lazy dependency is not supported for factories"
                )
            return cls._from_factory(ty.cast(IFactory[I, P], node), config)
