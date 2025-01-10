from abc import ABC
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from inspect import (
    BoundArguments,
    Parameter,
    Signature,
    isasyncgenfunction,
    isgeneratorfunction,
)
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    ForwardRef,
    Generator,
    Generic,
    Iterator,
    Union,
    cast,
    get_origin,
)

from typing_extensions import Unpack

from ._type_resolve import (
    IDIDI_IGNORE_PARAM_MARK,
    IDIDI_USE_FACTORY_MARK,
    FactoryType,
    ResolveOrder,
    flatten_annotated,
    get_args,
    get_typed_signature,
    is_any_context_manager_clss,
    is_any_generator,
    is_class,
    is_class_or_method,
    is_class_with_empty_init,
    is_unsolvable_type,
    resolve_annotation,
    resolve_factory_return,
    resolve_forwardref,
)
from .errors import (
    ABCNotImplementedError,
    MissingAnnotationError,
    NotSupportedError,
    ProtocolFacotryNotProvidedError,
)
from .interfaces import (
    EMPTY_SIGNATURE,
    INSPECT_EMPTY,
    INode,
    INodeAnyFactory,
    INodeConfig,
    INodeFactory,
    NodeIgnore,
    NodeIgnoreConfig,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T, get_factory_sig_from_cls

Ignore = Annotated[T, IDIDI_IGNORE_PARAM_MARK]


def use(
    factory: INodeFactory[P, T],
    **iconfig: Unpack[INodeConfig],
) -> T:
    """
    An annotation to let ididi knows what factory method to use
    without explicitly register it.

    These two are equivalent
    ```
    def func(service: UserService = use(factory)): ...
    def func(service: Annotated[UserService, use(factory)]): ...
    ```
    """
    node = DependentNode[T].from_node(factory, config=NodeConfig(**iconfig))
    annt = Annotated[node.dependent_type, node, IDIDI_USE_FACTORY_MARK]
    return cast(T, annt)


def search_meta(meta: list[Any]):
    for i, v in enumerate(meta):
        if v == IDIDI_USE_FACTORY_MARK:
            node: DependentNode[Any] = meta[i - 1]
            return node


def resolve_use(annotation: Any) -> Union["DependentNode[Any]", None]:
    if get_origin(annotation) is not Annotated:
        return

    meta: list[Any] = flatten_annotated(annotation)
    return search_meta(meta)


def should_override(
    other_node: "DependentNode[T]", current_node: "DependentNode[T]"
) -> bool:
    """
    Check if the other node should override the current node
    """

    ans = (
        ResolveOrder[other_node.factory_type] > ResolveOrder[current_node.factory_type]
    )
    return ans


class NodeConfig:
    __slots__ = ("reuse", "lazy", "ignore")

    ignore: NodeIgnore

    def __init__(
        self,
        *,
        reuse: bool = True,
        lazy: bool = False,
        ignore: Maybe[NodeIgnoreConfig] = MISSING,
    ):
        self.lazy = lazy

        if not is_provided(ignore):
            ignore = tuple()
        elif not isinstance(ignore, tuple):
            ignore = (ignore,)

        self.ignore = ignore
        self.reuse = reuse

    def __repr__(self):
        return f"{self.__class__.__name__}({self.reuse=}, {self.lazy=}, {self.ignore=})"


class Dependent(Generic[T]):
    """Represents a concrete (non-forward-reference) dependent type."""

    __slots__ = ("dependent_type",)

    def __init__(self, dependent_type: type[T]):
        self.dependent_type = dependent_type

    def __repr__(self) -> str:
        return self.dependent_name

    @property
    def dependent_name(self) -> str:
        return self.dependent_type.__name__


# NOTE: we might want to make this a descriptor
# for dpram in node.signature
#     setattr(node.dependent_type, LazyDescriptor)
class LazyDependent(Dependent[T]):
    __slots__ = (
        "dependent_type",
        "factory",
        "signature",
        "resolver",
        "cached_instance",
    )

    def __init__(
        self,
        *,
        dependent_type: type[T],
        factory: INodeFactory[P, T],
        signature: "DependentSignature[T]",
        resolver: Callable[[type[T]], T],
        cached_instance: Maybe[Union[T, Awaitable[T]]] = MISSING,
    ):
        self.dependent_type = dependent_type
        self.factory = factory
        self.signature = signature
        self.resolver = resolver
        self.cached_instance = cached_instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.dependent_type})"

    def __getattr__(self, attrname: str, /) -> Any:
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


@dataclass(frozen=True)
class DependencyParam(Generic[T]):
    """'dpram' for short
    Represents a parameter and its corresponding dependency node

    ```
    @dg.node
    def dependent_factory(self, n: int = 5) -> Any:
        ...
    ```

    Here 'n: str = 5' would be built into a DependencyParam

    name: the name of the param, here 'n' is the name
    param: inspect.Parameter, checkout inspect for this
    param_annotation: the original annotation of a by inspect, int, in this case.
    param_type: an ididi-friendly type resolved from annotation, int, in this case.
    default: the default value of the param, 5, in this case.
    """

    __slots__ = ("name", "param", "param_annotation", "param_type", "default")

    name: str
    param: Parameter
    param_annotation: type  # original type
    param_type: type[T]  # resolved_type
    default: Maybe[T]

    def __repr__(self) -> str:
        return f"{self.name}: {self.param_type}"

    @property
    def unresolvable(self) -> bool:
        "if the param is variadic or is of builtin without default"
        if self.param.kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True

        if self.default is MISSING and is_unsolvable_type(self.param_type):
            return True

        return False

    def replace_type(self, param_type: type[T]) -> "DependencyParam[T]":
        return DependencyParam(
            name=self.name,
            param=self.param,
            param_annotation=self.param_annotation,
            param_type=param_type,
            default=self.default,
        )

    def replace_default(self, default: Maybe[T]) -> "DependencyParam[T]":
        return DependencyParam(
            name=self.name,
            param=self.param,
            param_annotation=self.param_annotation,
            param_type=self.param_type,
            default=default,
        )


class DependentSignature(Generic[T]):
    __slots__ = ("dependent", "dprams")

    def __init__(
        self,
        *,
        dependent: type[Union[T, None]] = type(None),
        dprams: Union[dict[str, DependencyParam[Any]], None],
    ):
        self.dependent = dependent
        self.dprams = dprams or {}

    """
    # hash by dependent, and param names
    def __hash__(self):
        return hash((self.dependent, tuple(self.dprams.keys())))
    """

    def __iter__(self) -> Iterator[tuple[str, DependencyParam[Any]]]:
        return iter(self.dprams.items())

    def __len__(self) -> int:
        return len(self.dprams)

    def __getitem__(self, param_name: str) -> DependencyParam[Any]:
        return self.dprams[param_name]

    def generate_signature(self) -> Signature:
        """
        dynamically generate signature based on current params,
        as we might need to actualize forwardref params when bind params
        """
        return Signature(
            parameters=[p.param for p in self.dprams.values()],
            return_annotation=self.dependent,
            __validate_parameters__=False,
        )

    def bind_arguments(self, resolved_args: dict[str, Any]) -> BoundArguments:
        """Bind resolved arguments to the signature."""
        sig = self.generate_signature()
        bound_args = sig.bind_partial(**resolved_args)
        return bound_args

    @classmethod
    def from_signature(
        cls,
        dependent: type[T],
        factory: INodeFactory[P, T],
        signature: Signature,
    ):
        params = tuple(signature.parameters.values())

        if is_class_or_method(factory):
            params = params[1:]  # skip 'self'

        dep_params: dict[str, DependencyParam[Any]] = {}

        for param in params:
            param_annotation = param.annotation
            if param_annotation is INSPECT_EMPTY:
                e = MissingAnnotationError(
                    dependent_type=dependent,
                    param_name=param.name,
                    param_type=param_annotation,
                )
                e.add_context(dependent, param.name, type(MISSING))
                raise e

            if param.default == INSPECT_EMPTY:
                default = MISSING
            else:
                default = param.default

            dep_param = DependencyParam(
                name=param.name,
                param=param,
                param_annotation=param_annotation,
                param_type=resolve_annotation(param_annotation),
                default=default,
            )
            dep_params[param.name] = dep_param

        return cls(dprams=dep_params, dependent=dependent)


# ======================= Node =====================================


class DependentNode(Generic[T]):
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

    __slots__ = ("_dependent", "factory", "factory_type", "signature", "config")

    def __init__(
        self,
        *,
        dependent: Dependent[T],
        factory: INodeAnyFactory[T],
        factory_type: FactoryType,
        signature: DependentSignature[T],
        config: NodeConfig,
    ):
        self._dependent = dependent
        self.factory = factory
        self.factory_type: FactoryType = factory_type
        self.signature = signature
        self.config = config

    def __repr__(self) -> str:
        str_repr = f"{self.__class__.__name__}(type: {self._dependent}"
        if self.factory_type != "default":
            str_repr += f", factory: {self.factory.__qualname__}"
        str_repr += ")"
        return str_repr

    @property
    def dependent(self) -> Dependent[T]:
        return self._dependent

    @property
    def dependent_type(self) -> type[T]:
        return self._dependent.dependent_type

    @property
    def is_resource(self) -> bool:
        if self.factory_type == "resource":
            return True

        # class that implements __(a)enter__ and __(a)exit__
        return (
            is_any_context_manager_clss(self._dependent.dependent_type)
            and self.factory_type == "default"
        )

    def static_unsolved_params(self) -> Generator[DependencyParam[T], None, None]:
        "params that needs to be statically resolved"
        ignores = self.config.ignore
        for i, (param_name, param) in enumerate(self.signature):
            param_type = cast(Union[type, ForwardRef], param.param_type)
            if i in ignores or param_name in ignores or param_type in ignores:
                continue

            if get_origin(param_type) is Annotated:
                annotate_meta = flatten_annotated(param_type)
                if IDIDI_IGNORE_PARAM_MARK in annotate_meta:
                    self.config.ignore = ignores + (param_name,)
                    continue
                elif use_node := search_meta(annotate_meta):
                    yield param
                    param = param.replace_type(use_node.dependent_type)
                    self.signature.dprams[param_name] = param
                    continue
                else:
                    param_type, *_ = get_args(param_type)
                    param = param.replace_type(param_type)

            if is_provided(param.default) and get_origin(param.default) is Annotated:
                annt_default = param.default
                if use_node := resolve_use(annt_default):
                    param = param.replace_type(annt_default).replace_default(MISSING)
                    yield param
                    self.signature.dprams[param_name] = param.replace_type(
                        use_node.dependent_type
                    )
                    continue

            if isinstance(param_type, ForwardRef):
                param_type = resolve_forwardref(self.dependent_type, param_type)
                param = param.replace_type(param_type)

            yield param
            self.signature.dprams[param_name] = param

    def unsolved_params(
        self, ignore: NodeIgnore
    ) -> Generator[tuple[str, type], None, None]:
        "yield dependencies that needs to be resolved"
        ignores = self.config.ignore + ignore

        for i, (param_name, dpram) in enumerate(self.signature):
            param_type: type[T] = dpram.param_type

            if i in ignores or param_name in ignores or param_type in ignores:
                continue

            if is_provided(dpram.default) and is_unsolvable_type(param_type):
                continue

            yield (param_name, param_type)

    def inject_params(
        self, params: dict[str, Any]
    ) -> Union[T, Awaitable[T], Generator[T, None, None], AsyncGenerator[T, None]]:
        "Inject dependencies to the dependent accordidng to its signature, return an instance of the dependent type"
        arguments = self.signature.bind_arguments(params).arguments
        r = self.factory(**arguments)
        return r

    def check_for_resolvability(self) -> None:
        if isinstance(self.factory, type):
            # no factory override
            if getattr(self.factory, "_is_protocol", False):
                raise ProtocolFacotryNotProvidedError(self.factory)

            if issubclass(self.factory, ABC) and (
                abstract_methods := getattr(self.factory, "__abstractmethods__", None)
            ):
                raise ABCNotImplementedError(self.factory, abstract_methods)

    @classmethod
    def create(
        cls,
        *,
        dependent: type[T],
        factory: INodeFactory[P, T],
        factory_type: FactoryType,
        signature: Signature,
        config: NodeConfig,
    ) -> "DependentNode[T]":
        dep = Dependent(dependent_type=dependent)
        sig = DependentSignature[T].from_signature(dependent, factory, signature)
        node = DependentNode(
            dependent=dep,
            factory=factory,
            factory_type=factory_type,
            signature=sig,
            config=config,
        )
        return node

    @classmethod
    def from_factory(
        cls,
        *,
        factory: INodeFactory[P, T],
        config: NodeConfig,
    ) -> "DependentNode[T]":
        factory_type = "function"
        f = factory
        if isgeneratorfunction(f):
            f = contextmanager(f)
            factory_type = "resource"
        elif isasyncgenfunction(f):
            f = asynccontextmanager(f)
            factory_type = "resource"
        else:
            # case where user decorate a generator with contextlib.asynccontextmanager
            if is_any_generator(_ := getattr(factory, "__wrapped__", None)):
                factory_type = "resource"

        signature = get_typed_signature(f, check_return=True)
        dependent: type[T] = resolve_factory_return(signature.return_annotation)
        node = cls.create(
            dependent=dependent,
            factory=cast(INodeFactory[P, T], f),
            factory_type=factory_type,
            signature=signature,
            config=config,
        )
        return node

    @classmethod
    def from_class(
        cls, *, dependent: type[T], config: NodeConfig
    ) -> "DependentNode[T]":
        if hasattr(dependent, "__origin__"):
            if res := get_origin(dependent):
                dependent = res

        if is_class_with_empty_init(dependent):
            return cls.create(
                dependent=dependent,
                factory=dependent,
                factory_type="default",
                signature=EMPTY_SIGNATURE,
                config=config,
            )

        signature = get_factory_sig_from_cls(dependent)
        return cls.create(
            dependent=dependent,
            factory=dependent,
            factory_type="default",
            signature=signature,
            config=config,
        )

    @classmethod
    @lru_cache(maxsize=1000)
    def from_node(
        cls,
        factory_or_class: INode[P, T],
        *,
        config: Maybe[NodeConfig] = MISSING,
    ) -> "DependentNode[T]":
        """
        Build a node Nonrecursively
        from
            - class,
            - async / factory function of the class
        """
        if not is_provided(config):
            config = NodeConfig()

        if is_class(factory_or_class):
            dependent = cast(type[T], factory_or_class)
            return cls.from_class(dependent=dependent, config=config)
        else:
            if config.lazy:
                raise NotSupportedError(
                    "Lazy dependency is not supported for factories"
                )
            factory = cast(INodeFactory[P, T], factory_or_class)
            return cls.from_factory(factory=factory, config=config)
