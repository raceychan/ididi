from abc import ABC
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from inspect import Parameter, Signature
from inspect import _ParameterKind as ParameterKind  # type: ignore
from inspect import isasyncgenfunction, isgeneratorfunction
from typing import (
    Annotated,
    Any,
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
    get_factory_sig_from_cls,
    get_typed_signature,
    is_actxmgr_cls,
    is_class,
    is_class_or_method,
    is_class_with_empty_init,
    is_ctxmgr_cls,
    is_function,
    is_unsolvable_type,
    isasyncgenfunction,
    isgeneratorfunction,
    resolve_annotation,
    resolve_forwardref,
)
from .config import CacheMax, DefaultConfig, FrozenSlot, NodeConfig
from .errors import (
    ABCNotImplementedError,
    MissingAnnotationError,
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
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, T

# ============== Ididi marks ===========

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
    annt = Annotated[node.dependent, node, IDIDI_USE_FACTORY_MARK]
    return cast(T, annt)


# ============== Ididi marks ===========


def search_meta(meta: list[Any]) -> Union["DependentNode[Any]", None]:
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


# ======================= Signature =====================================


class Dependency(FrozenSlot, Generic[T]):
    """'dpram' for short
    Represents a parameter and its corresponding dependency node

    ```
    @dg.node
    def dependent_factory(self, n: int = 5) -> Any:
        ...
    ```

    Here 'n: str = 5' would be built into a DependencyParam

    name: the name of the param, here 'n' is the name
    param_type: an ididi-friendly type resolved from annotation, int, in this case.
    param_kind: inspect.ParameterKind
    default: the default value of the param, 5, in this case.
    """

    __slots__ = ("name", "param_type", "param_kind", "default")

    name: str
    param_type: Callable[..., T]  # resolved_type
    param_kind: ParameterKind
    default: Maybe[T]

    def __init__(
        self,
        name: str,
        param_type: Callable[..., T],
        param_kind: ParameterKind,
        default: Maybe[T],
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "param_type", param_type)
        object.__setattr__(self, "param_kind", param_kind)
        object.__setattr__(self, "default", default)

    def __repr__(self) -> str:
        return f"Dependency({self.name}: {self.type_repr}={self.default!r})"

    @property
    def type_repr(self):
        return getattr(self.param_type, "__name__", f"{self.param_type}")

    @property
    def unresolvable(self) -> bool:
        "if the param is variadic or is of builtin without default"
        if self.param_kind in (
            Parameter.VAR_POSITIONAL,
            Parameter.VAR_KEYWORD,
        ):
            return True

        if self.default is MISSING and is_unsolvable_type(self.param_type):
            return True

        return False

    def replace_type(self, param_type: type[T]) -> "Dependency[T]":
        return Dependency(
            name=self.name,
            param_kind=self.param_kind,
            param_type=param_type,
            default=self.default,
        )


def unpack_to_deps(
    param_annotation: Annotated[Any, "Unpack"]
) -> "dict[str, Dependency[Any]]":
    unpack: type = get_args(param_annotation)[0]
    fields = unpack.__annotations__
    dependencies: dict[str, Dependency[Any]] = {}

    for name, ftype in fields.items():
        dep_param = Dependency[Any](
            name=name,
            param_kind=ParameterKind.KEYWORD_ONLY,
            param_type=resolve_annotation(ftype),
            default=MISSING,
        )
        dependencies[name] = dep_param
    return dependencies


class Dependencies:
    __slots__ = ("_deps", "_sig")

    def __init__(
        self, deps: dict[str, Dependency[Any]], sig: Union[Signature, None] = None
    ):
        self._deps = deps
        self._sig = sig

    def __iter__(self) -> Iterator[tuple[str, Dependency[Any]]]:
        return iter(self._deps.items())

    def __len__(self) -> int:
        return len(self._deps)

    def __getitem__(self, param_name: str) -> Dependency[Any]:
        return self._deps[param_name]

    def __repr__(self):
        deps_repr = ", ".join(
            f"{name}: {dep.type_repr}={dep.default!r}" for name, dep in self
        )
        return f"Depenencies({deps_repr})"

    @property
    def signature(self) -> Signature:
        """
        dynamically generate signature based on current params,
        """

        if self._sig is None:
            self._sig = Signature(
                parameters=[
                    Parameter(
                        name=p.name,
                        kind=p.param_kind,
                        default=p.default,
                    )
                    for p in self._deps.values()
                ],
                __validate_parameters__=False,
            )
        return self._sig

    def update(self, param_name: str, param_val: Dependency[Any]):
        self._deps[param_name] = param_val

    def filter_ignore(
        self, ignore: NodeIgnore
    ) -> Generator[tuple[str, Dependency[Any]], None, None]:
        for param_name, param in self._deps.items():
            if (param_name in ignore) or (param.param_type in ignore):
                continue
            yield param_name, param

    @classmethod
    def from_signature(
        cls,
        factory: INodeFactory[P, T],
        signature: Signature,
        config: NodeConfig,
    ) -> "Dependencies":
        params = tuple(signature.parameters.values())
        if is_class_or_method(factory):
            params = params[1:]  # skip 'self'

        dependencies: dict[str, Dependency[Any]] = {}

        for param in params:
            param_annotation = param.annotation
            if param_annotation is INSPECT_EMPTY:
                e = MissingAnnotationError(
                    dependent_type=factory,
                    param_name=param.name,
                    param_type=param_annotation,
                )
                e.add_context(factory, param.name, type(MISSING))
                raise e

            param_default = param.default

            if param_default == INSPECT_EMPTY:
                default = MISSING
            else:
                default = param.default

            param_type = resolve_annotation(param_annotation)
            if get_origin(default) is Annotated:
                param_type = resolve_annotation(default)
                default = MISSING

            if get_origin(param_type) is Annotated:
                annotate_meta = flatten_annotated(param_type)
                if IDIDI_IGNORE_PARAM_MARK in annotate_meta:
                    continue

                if IDIDI_USE_FACTORY_MARK not in annotate_meta:
                    param_type, *rest = get_args(param_type)
                    for r in rest:
                        if is_function(r):
                            param_type = r

            if param_type is Unpack:
                dependencies.update(unpack_to_deps(param_annotation))
                continue

            dep_param = Dependency(
                name=param.name,
                param_kind=param.kind,
                param_type=param_type,
                default=default,
            )
            dependencies[param.name] = dep_param

        ignore = config.ignore
        excluded_deps = {
            param_name: param
            for pos_idx, (param_name, param) in enumerate(dependencies.items())
            if (
                pos_idx not in ignore
                and param_name not in ignore
                and param.param_type not in ignore
            )
        }
        return cls(excluded_deps)


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

    reuse: bool
    ---
    whether this node is reusable, default is True
    """

    __slots__ = (
        "dependent",
        "factory",
        "factory_type",
        "function_dependent",
        "dependencies",
        "config",
    )

    def __init__(
        self,
        *,
        dependent: Callable[..., T],
        factory: INodeAnyFactory[T],
        factory_type: FactoryType,
        function_dependent: bool = False,
        dependencies: Dependencies,
        config: NodeConfig,
    ):

        self.dependent = dependent
        self.factory = factory
        self.factory_type: FactoryType = factory_type
        self.function_dependent = function_dependent
        self.dependencies = dependencies
        self.config = config

    def __repr__(self) -> str:
        str_repr = f"{self.__class__.__name__}(type: {self.dependent}"
        if self.factory_type != "default":
            str_repr += f", factory: {self.factory}"
        str_repr += ")"
        return str_repr

    @property
    def is_resource(self) -> bool:
        return self.factory_type in ("resource", "aresource")

    def analyze_unsolved_params(
        self, ignore: NodeIgnore
    ) -> Generator[Dependency[T], None, None]:
        "params that needs to be statically resolved"

        for param_name, param in self.dependencies.filter_ignore(ignore):
            param_type = cast(Union[Callable[..., T], ForwardRef], param.param_type)
            if isinstance(param_type, ForwardRef):
                new_type = resolve_forwardref(self.dependent, param_type)
                param = param.replace_type(new_type)
            yield param
            if param.param_type != param_type:
                self.dependencies.update(param_name, param)

    @lru_cache(CacheMax)
    def unsolved_params(self, ignore: NodeIgnore) -> list[tuple[str, Callable[..., T]]]:
        "yield dependencies that needs to be resolved"
        unsolved_params: list[tuple[str, Callable[..., T]]] = []
        for param_name, param in self.dependencies.filter_ignore(ignore=ignore):
            param_type: Callable[..., T] = param.param_type
            if is_provided(param.default) and is_unsolvable_type(param_type):
                continue
            param_pair = (param_name, param_type)
            unsolved_params.append(param_pair)
        return unsolved_params

    def check_for_implementations(self) -> None:
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
        dependent_type: type[T],
        factory: INodeFactory[P, T],
        factory_type: FactoryType,
        signature: Signature,
        config: NodeConfig,
    ) -> "DependentNode[T]":
        dependencies = Dependencies.from_signature(
            factory=factory, signature=signature, config=config
        )
        node = DependentNode(
            dependent=resolve_annotation(dependent_type),
            factory=factory,
            factory_type=factory_type,
            dependencies=dependencies,
            config=config,
        )
        return node

    @classmethod
    def _from_factory(
        cls,
        *,
        factory: INodeFactory[P, T],
        config: NodeConfig,
    ) -> "DependentNode[T]":

        f = factory
        factory_type: FactoryType
        if isasyncgenfunction(factory):
            f = asynccontextmanager(factory)
            factory_type = "aresource"
        elif isgeneratorfunction(factory):
            f = contextmanager(factory)
            factory_type = "resource"
        elif genfunc := getattr(factory, "__wrapped__", None):
            factory_type = "aresource" if isasyncgenfunction(genfunc) else "resource"
        else:
            factory_type = "function"

        signature = get_typed_signature(f, check_return=True)
        dependent: type[T] = resolve_annotation(signature.return_annotation)

        if get_origin(dependent) is Annotated:
            metas = flatten_annotated(dependent)
            if IDIDI_IGNORE_PARAM_MARK in metas:
                node = cls._from_function(f, factory_type=factory_type, config=config)
                return node

        node = cls.create(
            dependent_type=dependent,
            factory=cast(INodeFactory[P, T], f),
            factory_type=factory_type,
            signature=signature,
            config=config,
        )
        return node

    @classmethod
    def _from_class(
        cls, *, dependent: type[T], config: NodeConfig
    ) -> "DependentNode[T]":
        if is_class_with_empty_init(dependent):
            signature = EMPTY_SIGNATURE
        else:
            signature = get_factory_sig_from_cls(dependent)

        if is_ctxmgr_cls(dependent):
            factory_type = "resource"
        elif is_actxmgr_cls(dependent):
            factory_type = "aresource"
        else:
            factory_type = "default"

        return cls.create(
            dependent_type=dependent,
            factory=dependent,
            factory_type=factory_type,
            signature=signature,
            config=config,
        )

    @classmethod
    @lru_cache(CacheMax)
    def from_node(
        cls,
        factory_or_class: INode[P, T],
        *,
        config: NodeConfig = DefaultConfig,
    ) -> "DependentNode[T]":
        """
        Build a node Nonrecursively
        from
            - class,
            - async / factory function of the class
        """

        if is_class(factory_or_class):
            dependent = cast(type[T], factory_or_class)
            return cls._from_class(dependent=dependent, config=config)
        else:
            factory = cast(INodeFactory[P, T], factory_or_class)
            return cls._from_factory(factory=factory, config=config)

    @classmethod
    def _from_function(
        cls,
        function: Any,
        *,
        factory_type: FactoryType,
        config: NodeConfig = DefaultConfig,
    ):
        deps = Dependencies.from_signature(
            function, get_typed_signature(function), config
        )

        node = DependentNode(
            dependent=function,
            factory=function,
            factory_type=factory_type,
            function_dependent=True,
            dependencies=deps,
            config=config,
        )
        return node
