from abc import ABC
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from inspect import Signature
from inspect import _ParameterKind as ParameterKind  # type: ignore
from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction
from types import FunctionType, MethodType
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    ForwardRef,
    Generator,
    Union,
    cast,
    get_origin,
    overload,
)

from typing_extensions import Unpack

from ._type_resolve import (
    flatten_annotated,
    get_args,
    get_factory_sig_from_cls,
    get_typed_signature,
    is_actxmgr_cls,
    is_class,
    is_class_with_empty_init,
    is_ctxmgr_cls,
    is_unsolvable_type,
    isasyncgenfunction,
    isgeneratorfunction,
    resolve_annotation,
    resolve_forwardref,
)
from .config import (
    IGNORE_PARAM_MARK,
    USE_FACTORY_MARK,
    CacheMax,
    EmptyIgnore,
    FactoryType,
    ResolveOrder,
)
from .errors import (
    ABCNotImplementedError,
    DeprecatedError,
    MissingAnnotationError,
    ProtocolFacotryNotProvidedError,
)
from .interfaces import (
    EMPTY_SIGNATURE,
    INSPECT_EMPTY,
    IDependent,
    INode,
    INodeAnyFactory,
    INodeFactory,
    NodeIgnoreConfig,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, R, T

# ============== Ididi marks ===========

Ignore = Annotated[R, IGNORE_PARAM_MARK]

# ========== NotImplemented =======

Scoped = Annotated[Union[Generator[T, None, None], AsyncGenerator[T, None]], "scoped"]

# ========== NotImplemented =======

@overload
def use(*, reuse: bool) -> Any: ...

@overload
def use(func: Maybe[INode[P, T]], /, *, reuse: bool = False) -> T: ...

def use(func: Maybe[INode[P, T]] = MISSING, /, *,  reuse: bool = False) -> T:
    """
    An annotation to let ididi knows what factory method to use
    without explicitly register it.

    These two are equivalent
    ```
    def func(service: Annotated[UserService, use(factory)]): ...
    ```
    """

    annt = Annotated[T, USE_FACTORY_MARK, func, reuse]
    return cast(T, annt)



# ============== Ididi marks ===========


def _search_meta(meta: list[Any]) -> Union[tuple[IDependent[Any], bool], None]:
    for i, v in enumerate(meta):
        if v == USE_FACTORY_MARK:
            func, reuse = meta[i + 1], meta[i + 2]
            return func, reuse
    return None


def resolve_use(annotation: Any) -> Union[tuple[Union[IDependent[Any], None], bool], None]:
    if get_origin(annotation) is not Annotated:
        return

    metas: list[Any] = flatten_annotated(annotation)
    use_meta = _search_meta(metas)
    if use_meta is None:
        return None
    
    factory, reuse = use_meta
    if not is_provided(factory):
        factory = get_args(annotation)[0]
    return factory, reuse


def should_override(other_node: "DependentNode", current_node: "DependentNode") -> bool:
    """
    Check if the other node should override the current node
    """

    other_priority = ResolveOrder[other_node.factory_type]
    current_priority = ResolveOrder[current_node.factory_type]
    return  other_priority > current_priority

def resolve_type_from_meta(annt: Any) -> IDependent[Any]:
    """
    resolve types from Annotated[T, ...]
    such as 
    user_serivce: Annotated[UserService, ...]

    if use mark or ignore mark in meta, don't handle, leave it to later
    otherwise, meaning that it is something user leave for them self, so return first param_type 

    """
    annotate_meta = flatten_annotated(annt)

    if IGNORE_PARAM_MARK in annotate_meta:
        return annt
    if USE_FACTORY_MARK in annotate_meta:
        return annt
    return get_args(annt)[0]


# ======================= Signature =====================================


# TODO: make this a cython struct
class Dependency:
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

    name: str
    param_type: "IDependent[Any]"  # resolved_type
    default_: "Maybe[Any]"
    should_be_ignored: bool

    def __init__(
        self,
        name: str,
        param_type: IDependent[T],
        default: Maybe[T],
    ) -> None:
        self.name = name
        self.param_type = param_type
        self.default_ = default
        self.should_be_ignored = self.should_ignore(param_type) or is_unsolvable_type(
            param_type
        )

    def should_ignore(self, param_type: IDependent[T]) -> bool:
        return get_origin(
            param_type
        ) is Annotated and IGNORE_PARAM_MARK in flatten_annotated(param_type)

    def __repr__(self) -> str:
        return f"Dependency({self.name}: {self.type_repr}={self.default_!r})"

    @property
    def type_repr(self):
        return self.param_type

    def replace_type(self, param_type: IDependent[T]) -> "Dependency":
        return Dependency(
            name=self.name,
            param_type=param_type,
            default=self.default_,
        )


def unpack_to_deps(
    param_annotation: Annotated[Any, "Unpack"],
) -> "dict[str, Dependency]":
    unpack = get_args(param_annotation)[0]
    fields = unpack.__annotations__
    dependencies: dict[str, Dependency] = {}

    for name, ftype in fields.items():
        dep_param = Dependency(
            name=name,
            param_type=resolve_annotation(ftype),
            default=MISSING,
        )
        dependencies[name] = dep_param
    return dependencies


# TODO: make this an immutable dict that is faster to access
# all mutation only happens at analyze time, this will boost resolve.
class Dependencies(dict[str, Dependency]):
    __slots__ = "_deps"

    def __repr__(self):
        deps_repr = ", ".join(
            f"{name}: {dep.type_repr}={dep.default_!r}" for name, dep in self.items()
        )
        return f"Depenencies({deps_repr})"

    @classmethod
    def from_signature(
        cls,
        function: INodeFactory[P, T],
        signature: Signature,
    ) -> "Dependencies":
        params = tuple(signature.parameters.values())

        if isinstance(function, type):
            params = params[1:]  # skip 'self', 'cls'
        elif isinstance(function, MethodType):
            owner = function.__self__
            _ = getattr(owner, function.__name__)
            if not isinstance(owner, type):
                params = params[1:]  # skip 'self', 'cls'

        dependencies: dict[str, Dependency] = {}

        for param in params:
            param_annotation = param.annotation
            if param_annotation is INSPECT_EMPTY:
                e = MissingAnnotationError(
                    dependent_type=function,
                    param_name=param.name,
                    param_type=param_annotation,
                )
                e.add_context(function, param.name, type(MISSING))
                raise e

            param_default = param.default

            if param_default == INSPECT_EMPTY:
                default = MISSING
            else:
                default = param.default

            param_type = resolve_annotation(param_annotation)

            if get_origin(default) is Annotated:
                if resolve_use(default):
                    raise DeprecatedError(f"Using default value `{param}` for `use` is no longer supported")

            if get_origin(param_type) is Annotated:
                param_type = resolve_type_from_meta(param_type)

            if param_type is Unpack:
                dependencies.update(unpack_to_deps(param_annotation))
                continue

            if param.kind in (ParameterKind.VAR_POSITIONAL, ParameterKind.VAR_KEYWORD):
                continue

            dep_param = Dependency(
                name=param.name,
                param_type=param_type,
                default=default,
            )
            dependencies[param.name] = dep_param

        return cls(dependencies)


# ======================= Node =====================================


class DependentNode:
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

    def __init__(
        self,
        *,
        dependent: IDependent[T],
        factory: INodeAnyFactory[T],
        factory_type: FactoryType,
        function_dependent: bool = False,
        dependencies: Dependencies,
        reuse: bool, 
        ignore: NodeIgnoreConfig, 
        
    ):
        self.dependent = dependent
        self.factory = factory
        self.factory_type: FactoryType = factory_type
        self.function_dependent = function_dependent
        self.dependencies = dependencies
        self._reuse = reuse
        self._ignore = ignore

    @property
    def reuse(self):
        return self._reuse

    @property
    def ignore(self):
        return self._ignore

    def __repr__(self) -> str:
        str_repr = f"{self.__class__.__name__}(type: {self.dependent}"
        if self.factory_type != "default":
            str_repr += f", factory: {self.factory}"
        str_repr += f", function_dependent: {self.function_dependent}, reuse: {self._reuse}"
        str_repr += ")"
        return str_repr

    @property
    def is_resource(self) -> bool:
        return self.factory_type in ("resource", "aresource")

    def analyze_unsolved_params(
        self, ignore: tuple[Any, ...] = EmptyIgnore
    ) -> Generator[Dependency, None, None]:
        "params that needs to be statically resolved"
        ignore += self._ignore # type: ignore

        for i, (name, param) in enumerate(self.dependencies.items()):
            if i in ignore or name in ignore:
                continue
            param_type = cast(Union[IDependent[object], ForwardRef], param.param_type)
            if param_type in ignore:
                continue
            if isinstance(param_type, ForwardRef):
                new_type = resolve_forwardref(self.dependent, param_type)
                param = param.replace_type(new_type)
            yield param
            if param.param_type != param_type:
                self.dependencies[name] = param

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
        dependent: IDependent[T],
        factory: INodeFactory[P, T],
        factory_type: FactoryType,
        signature: Signature,
        reuse: bool, 
        ignore: NodeIgnoreConfig,
    ) -> "DependentNode":

        dependencies = Dependencies.from_signature(
            function=factory, signature=signature
        )
        node = DependentNode(
            dependent=resolve_annotation(dependent),
            factory=factory,
            factory_type=factory_type,
            dependencies=dependencies,
            reuse=reuse,
            ignore=ignore
        )
        return node

    @classmethod
    def _from_function_dep(
        cls,
        function: Any,
        *,
        signature: Signature,
        factory_type: FactoryType,
        reuse: bool,
        ignore: NodeIgnoreConfig,
    ):
        deps = Dependencies.from_signature(function=function, signature=signature)
        node = DependentNode(
            dependent=function,
            factory=function,
            factory_type=factory_type,
            function_dependent=True,
            dependencies=deps,
            reuse=reuse,
            ignore=ignore
        )
        return node

    @classmethod
    def _from_function(
        cls,
        *,
        factory: INodeFactory[P, T],
        reuse: bool,
        ignore: NodeIgnoreConfig
    ) -> "DependentNode":

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
        elif iscoroutinefunction(factory):
            factory_type = "afunction"
        else:
            factory_type = "function"
        signature = get_typed_signature(f, check_return=True)
        dependent: type[T] = resolve_annotation(signature.return_annotation)


        if get_origin(dependent) is Annotated:
            metas = flatten_annotated(dependent)
            if IGNORE_PARAM_MARK in metas:
                node = cls._from_function_dep(
                    f, signature=signature, factory_type=factory_type, reuse=reuse, ignore=ignore
                )
                return node

            base_dependent, *_ = get_args(dependent)
            use_meta = resolve_use(dependent)
            if use_meta:
                use_factory, use_reuse = use_meta
                if use_reuse:
                    reuse = True
                if use_factory not in (base_dependent, factory):
                    return cls.from_node(use_factory, reuse=reuse, ignore=ignore)

            dependent = base_dependent

        node = cls.create(
            dependent=dependent,
            factory=cast(INodeFactory[P, T], f),
            factory_type=factory_type,
            signature=signature,
            reuse=reuse,
            ignore=ignore
        )
        return node

    @classmethod
    def _from_class(cls, *, dependent: type[T], reuse: bool, ignore: NodeIgnoreConfig) -> "DependentNode":
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
            dependent=dependent,
            factory=dependent,
            factory_type=factory_type,
            signature=signature,
            reuse=reuse,
            ignore=ignore
        )

    @classmethod
    @lru_cache(CacheMax)
    def from_node(
        cls,
        factory_or_class: INode[P, T],
        *,
        reuse: bool = False,
        ignore: NodeIgnoreConfig = EmptyIgnore,
    ) -> "DependentNode":
        """
        Build a node Nonrecursively
        from
            - class,
            - async / factory function of the class
        """
        if not is_provided(ignore):
            ignore = tuple[Any, ...]()

        if is_class(factory_or_class):
            dependent = cast(type, factory_or_class)
            return cls._from_class(dependent=dependent, reuse=reuse,ignore=ignore)
        elif isinstance(factory_or_class, (FunctionType, MethodType)):
            factory = cast(INodeFactory[P, T], factory_or_class)
            return cls._from_function(factory=factory, reuse=reuse, ignore=ignore)
        else:
            raise TypeError(f"Invalid dependent type {factory_or_class}")
