import abc
import inspect
import typing as ty
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import Parameter

from ._itypes import EMPTY_SIGNATURE, INSPECT_EMPTY, IAsyncFactory, IFactory, NodeConfig
from ._type_resolve import (
    get_sig_origin_return,
    get_typed_signature,
    is_class,
    is_class_or_method,
    is_class_with_empty_init,
    is_unresolved_type,
    resolve_annotation,
)
from .errors import (
    ABCNotImplementedError,
    MissingAnnotationError,
    NotSupportedError,
    ProtocolFacotryNotProvidedError,
    UnsolvableDependencyError,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import get_factory_sig_from_cls


@dataclass(slots=True)
class Dependent[T]:
    """Represents a concrete (non-forward-reference) dependent type."""

    dependent_type: type[T]

    @property
    def dependent_name(self) -> str:
        return self.dependent_type.__name__


# we might make this a descriptor
# for dpram in node.signature
#     setattr(node.dependent_type, LazyDescriptor)
@dataclass(slots=True)
class LazyDependent[T](Dependent[T]):
    dependent_type: type[T]
    factory: IFactory[..., T] | IAsyncFactory[..., T]
    signature: "DependentSignature[T]"
    resolver: ty.Callable[[type[T]], T]
    cached_instance: Maybe[T | ty.Awaitable[T]] = MISSING

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
            return dpram.param_type
        except KeyError:
            if self.cached_instance is MISSING:
                self.cached_instance = self.resolver(self.dependent_type)
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
    param: Parameter  # TODO: do no rely on inspect.Parameter
    param_annotation: type  # original type
    param_type: type[T]  # resolved_type
    default: Maybe[T]

    def __repr__(self) -> str:
        return f"{self.param.name}: {self.param_type}"

    @property
    def unresolvable(self) -> bool:
        if self.param.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True

        if not is_provided(self.default) and is_unresolved_type(self.param_type):
            return True

        return False


@dataclass(slots=True, frozen=True, kw_only=True)
class DependentSignature[T]:
    dependent: type[T | None] = field(default=type(None))
    dprams: dict[str, DependencyParam[ty.Any]] = field(default_factory=dict, repr=False)

    """
    # hash by dependent, and param names
    def __hash__(self):
        return hash((self.dependent, tuple(self.dprams.keys())))
    """

    def __iter__(self) -> ty.Iterator[DependencyParam[ty.Any]]:
        return iter(self.dprams.values())

    def __len__(self) -> int:
        return len(self.dprams)

    def __getitem__(self, param_name: str) -> DependencyParam[ty.Any]:
        return self.dprams[param_name]

    def update(self, param_name: str, dpram: DependencyParam[ty.Any]) -> None:
        self.dprams[param_name] = dpram

    def get_signature(self) -> inspect.Signature:
        return inspect.Signature(
            parameters=[dpram.param for dpram in self.dprams.values()],
            return_annotation=self.dependent,
        )

    def bind_arguments(
        self, resolved_args: dict[str, ty.Any]
    ) -> inspect.BoundArguments:
        """Bind resolved arguments to the signature."""
        sig = self.get_signature()
        bound_args = sig.bind_partial(**resolved_args)
        bound_args.apply_defaults()
        return bound_args


# ======================= Node =====================================


def _create_sig[
    T
](
    *, dependent: type[T], params: ty.Sequence[Parameter], config: NodeConfig
) -> DependentSignature[T]:
    """
    Create a signature for a dependent type.
    """

    dep_params: dict[str, DependencyParam[ty.Any]] = {}

    for param in params:
        param_annotation = param.annotation
        if param_annotation is INSPECT_EMPTY:
            raise MissingAnnotationError(param.name, dependent)

        if param.default == INSPECT_EMPTY:
            default = MISSING
        else:
            default = param.default

        dep_param = DependencyParam(
            param=param,
            name=param.name,
            param_annotation=param_annotation,
            param_type=resolve_annotation(param_annotation),
            default=default,
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

    You can also register a factory async/sync function that returns the dependent type as a node,
    e.g.
    ```python
    @dg.node
    def auth_service_factory() -> AuthService:
        return AuthService(redis, keyspace)
    ```
    in this case:
    - dependent: the dependent type that this node represents, e.g AuthService
    - factory: the factory function that creates the dependent, e.g auth_service_factory

    ## [config]

    lazy: bool
    ---
    whether this node is lazy, default is False

    reuse: bool
    ---
    whether this node is reusable, default is True
    """

    dependent: Dependent[T]
    factory: IFactory[..., T] | IAsyncFactory[..., T]
    default: Maybe[T] = MISSING
    signature: DependentSignature[T]
    config: NodeConfig

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dependent})"

    @property
    def dependent_type(self) -> type[T]:
        return self.dependent.dependent_type

    def check_for_resolvability(self) -> None:
        if isinstance(self.factory, type):
            # no factory override
            if is_unresolved_type(self.factory):
                raise UnsolvableDependencyError(
                    dep_name=self.dependent.dependent_name,
                    required_type=self.dependent_type,
                    factory=self.factory,
                )

            if getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)

            if issubclass(self.factory, abc.ABC) and (
                abstract_methods := getattr(self.factory, "__abstractmethods__", None)
            ):
                raise ABCNotImplementedError(self.factory, abstract_methods)

        for dpram in self.signature:
            if dpram.unresolvable:
                raise UnsolvableDependencyError(
                    dep_name=dpram.param.name,
                    factory=dpram.param.annotation,
                    required_type=self.dependent_type,
                )

    @classmethod
    def create[
        **P, R
    ](
        cls,
        *,
        dependent: type[R],
        factory: IFactory[P, R] | IAsyncFactory[P, R],
        signature: inspect.Signature,
        config: NodeConfig,
    ) -> "DependentNode[R]":
        params = tuple(signature.parameters.values())

        if is_class_or_method(factory):
            params = params[1:]  # skip 'self'

        dpram_sig = _create_sig(dependent=dependent, params=params, config=config)

        dep = Dependent(dependent_type=dependent)
        node = DependentNode(
            dependent=dep,
            factory=factory,
            signature=dpram_sig,
            config=config,
        )
        return node

    @classmethod
    def from_factory[
        I, **P
    ](
        cls, *, factory: IFactory[P, I] | IAsyncFactory[P, I], config: NodeConfig
    ) -> "DependentNode[I]":
        signature = get_typed_signature(factory, check_return=True)
        dependent: type[I] = get_sig_origin_return(signature.return_annotation)
        node = cls.create(
            dependent=dependent, factory=factory, signature=signature, config=config
        )
        return node

    @classmethod
    def _create_empty_init_node(
        cls, dependent: type[ty.Any], config: NodeConfig
    ) -> "DependentNode[ty.Any]":
        return cls.create(
            dependent=dependent,
            factory=dependent,
            signature=EMPTY_SIGNATURE,
            config=config,
        )

    @classmethod
    def from_class[
        I
    ](cls, *, dependent: type[I], config: NodeConfig) -> "DependentNode[I]":
        if hasattr(dependent, "__origin__"):
            if res := ty.get_origin(dependent):
                dependent = res

        if is_class_with_empty_init(dependent):
            return cls._create_empty_init_node(dependent, config)

        signature = get_factory_sig_from_cls(dependent)
        return cls.create(
            dependent=dependent, factory=dependent, signature=signature, config=config
        )

    @classmethod
    @lru_cache(maxsize=1000)
    def from_node[
        I, **P
    ](
        cls,
        factory_or_class: type[I] | IFactory[P, I] | IAsyncFactory[P, I],
        *,
        config: Maybe[NodeConfig] = MISSING,
    ) -> "DependentNode[I]":
        """
        Build a node from a class, a factory function of the class, or an async factory function of the class
        """
        if not is_provided(config):
            config = NodeConfig()

        if is_class(factory_or_class):
            dependent = factory_or_class
            return cls.from_class(dependent=dependent, config=config)
        else:
            # TODO: support lazy factory
            # where attributes other than keyword arguments are lazy
            if config.lazy:
                raise NotSupportedError(
                    "Lazy dependency is not supported for factories"
                )
            factory = factory_or_class
            return cls.from_factory(factory=factory, config=config)
