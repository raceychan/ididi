from abc import ABC
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import lru_cache
from inspect import Signature
from inspect import _ParameterKind as ParameterKind  # type: ignore
from inspect import isasyncgenfunction, iscoroutinefunction, isgeneratorfunction
from types import FunctionType, MethodType
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Callable,
    ForwardRef,
    Generator,
    Generic,
    Union,
    cast,
    get_origin,
    overload,
)

from typing_extensions import Unpack

from ._type_resolve import (
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
from .config import CacheMax, EmptyIgnore, FactoryType, ResolveOrder
from .errors import (
    ABCNotImplementedError,
    ConfigConflictError,
    MissingAnnotationError,
    NotSupportedError,
    ProtocolFacotryNotProvidedError,
)
from .interfaces import (
    EMPTY_SIGNATURE,
    INSPECT_EMPTY,
    IDependent,
    INode,
    INodeAnyFactory,
    INodeFactory,
    NodeIgnore,
    NodeIgnoreConfig,
)
from .utils.param_utils import MISSING, Maybe, is_provided
from .utils.typing_utils import P, R, T, flatten_annotated

# ============== Ididi marks ===========

Scoped = Annotated[Union[Generator[T, None, None], AsyncGenerator[T, None]], "scoped"]


@dataclass(frozen=True)
class NodeMeta(Generic[T]):
    factory: Maybe[Callable[..., T]] = MISSING
    reuse: Maybe[bool] = MISSING
    ignore: bool = False

    def __post_init__(self):
        if self.ignore and (self.factory or self.reuse):
            raise NotSupportedError(f"Ignored node can't have factory or being reused")

Ignore = Annotated[R, NodeMeta(ignore=True)]



@overload
def use(*, reuse: Maybe[bool]=MISSING) -> Any: ...

@overload
def use(factory: Maybe[INode[P, T]], /, *, reuse: Maybe[bool] = MISSING) -> T: ...

def use(factory: Maybe[INode[P, T]] = MISSING, /, *,  reuse: Maybe[bool] = MISSING) -> T:
    """
    An annotation to let ididi knows what factory method to use
    without explicitly register it.

    These two are equivalent
    ```
    def func(service: Annotated[UserService, use(factory)]): ...
    def func(service: Annotated[UserService, use()]): ...
    ```
    """

    annt = Annotated[T, NodeMeta(factory=factory, reuse=reuse)]
    return cast(T, annt)



# ============== Ididi marks ===========




def resolve_meta(annotation: Any) -> Union[NodeMeta[Any], None]:
    "Accept any type hint or a list of elements"
    if isinstance(annotation, list):
        metas: list[Any] = annotation
        annt_args = None
    elif get_origin(annotation) is not Annotated:
        return
    else:
        metas: list[Any] = flatten_annotated(annotation)
        annt_args = get_args(annotation)

    for v in metas:
        if isinstance(v, NodeMeta):
            if v.ignore is True: # we should not care about its factory if its ignored
                return v
            factory = annt_args[0] if (not is_provided(v.factory) and annt_args) else v.factory
            return NodeMeta(factory=factory, reuse=v.reuse, ignore=v.ignore)
    return None



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

    if any(isinstance(meta, NodeMeta) for meta in annotate_meta):
        return annt
    return get_args(annt)[0]


# ======================= Signature =====================================


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

    __slots__ = ("name", "param_type", "annotation", "default_", "should_be_ignored")

    def __init__(
        self,
        name: str,
        param_type: IDependent[T],
        annotation: Any,
        default: Maybe[T],
    ) -> None:
        self.name = name
        self.param_type = param_type
        self.annotation = annotation
        self.default_ = default
        self.should_be_ignored = self.should_ignore(param_type) or is_unsolvable_type(
            param_type
        )

    def __repr__(self) -> str:
        return f"Dependency({self.name}: {self.type_repr}={self.default_!r})"

    def should_ignore(self, param_type: IDependent[T]) -> bool:
        meta = resolve_meta(param_type)
        if not is_provided(meta) or meta is None:
            return False
        return meta.ignore

    @property
    def type_repr(self):
        return self.param_type

    def replace_type(self, param_type: IDependent[T]) -> "Dependency":
        return Dependency(
            name=self.name,
            param_type=param_type,
            annotation=self.annotation,
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
            annotation=ftype,
            default=MISSING,
        )
        dependencies[name] = dep_param
    return dependencies


# TODO: make this an immutable dict that is faster to access
# all mutation only happens at analyze time, this will boost resolve.

def build_dependencies(
    function: INodeFactory[P, T],
    signature: Signature,
) -> list[Dependency]:
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
            annotation=param_annotation,
            default=default,
        )
        dependencies[param.name] = dep_param

    dep_values = list(dependencies.values())
    return dep_values


# ======================= Node =====================================

"""
TODO:
1. remove class Dependencies, just use plain dict
2. move DependentNode.reuse to Dependency
3. Rename Dependency to Edge
4. Virtual Edge Vs mature edge

class EdgeBase:
    name: str
    reuse: bool

class VirtualEdge(EdgeBase):
    type: type | ForwardRef


class MatureEdge(EdgeBase):
    type: DependentNode
"""

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

    __slots__ = (
        "dependent",
        "factory",
        "factory_type",
        "function_dependent",
        "dependencies",
        "_reuse",
        "_ignore",
    )

    dependencies: list[Dependency]

    def __init__(
        self,
        *,
        dependent: IDependent[T],
        factory: INodeAnyFactory[T],
        factory_type: FactoryType,
        function_dependent: bool = False,
        signature: Maybe[Signature] = MISSING,
        reuse: Maybe[bool] = MISSING, 
        ignore: NodeIgnore
    ):
        self.dependent = dependent
        self.factory = factory
        self.factory_type: FactoryType = factory_type
        self.function_dependent = function_dependent
        if not is_provided(signature):
            self.dependencies = []
        else:
            self.dependencies = build_dependencies(factory, signature=signature)
        self._reuse = reuse
        self._ignore = ignore

    @property
    def reuse(self) -> Maybe[bool]:
        return self._reuse

    @property
    def ignore(self):
        return self._ignore

    def get_param(self, name: str) -> Dependency:
        for p in self.dependencies:
            if p.name == name:
                return p
        raise AttributeError(f"{name} not found")

    def update_reusability(self, other: "DependentNode") -> None:
        if not is_provided(other._reuse):
            return 

        if not is_provided(self._reuse):
            self._reuse = other._reuse
        elif other._reuse != self._reuse:
            raise ConfigConflictError(self.factory, self._reuse, other.factory, other._reuse)



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

    def update_param(self, name: str, new_type: IDependent[Any])->None:
        for i, param in enumerate(self.dependencies):
            if param.name != name:
                continue
            new_param = param.replace_type(new_type)
            self.dependencies[i] = new_param

    def analyze_unsolved_params(
        self, ignore: tuple[Any, ...] = EmptyIgnore
    ) -> Generator[Dependency, None, None]:
        "params that needs to be statically resolved"
        ignore += self._ignore # type: ignore

        for i, param in enumerate(self.dependencies):
            name = param.name
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
                self.dependencies[i] = param

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
        reuse: Maybe[bool], 
        ignore: NodeIgnore,
    ) -> "DependentNode":
        node = DependentNode(
            dependent=resolve_annotation(dependent),
            factory=factory,
            signature=signature,
            factory_type=factory_type,
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
        reuse: Maybe[bool],
        ignore: NodeIgnore,
    ):
        node = DependentNode(
            dependent=function,
            factory=function,
            signature=signature,
            factory_type=factory_type,
            function_dependent=True,
            reuse=reuse,
            ignore=ignore
        )
        return node

    @classmethod
    @lru_cache(CacheMax)
    def _from_function(
        cls,
        factory: INodeFactory[P, T],
        *,
        reuse: Maybe[bool],
        ignore: NodeIgnore
    ) -> "DependentNode":

        f = factory
        factory_type: FactoryType
        if isasyncgenfunction(factory):
            f = asynccontextmanager(func=factory)
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
            node_meta = resolve_meta(dependent)
            if node_meta and node_meta.ignore:
                node = cls._from_function_dep(
                    f, signature=signature, factory_type=factory_type, reuse=reuse, ignore=ignore
                )
                return node

            base_dependent, *_ = get_args(dependent)
            if node_meta := resolve_meta(dependent):
                if not is_provided(reuse):
                    reuse = node_meta.reuse

                if node_meta.factory not in (base_dependent, factory):
                    return cls.from_node(node_meta.factory, reuse=reuse, ignore=ignore)

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
    @lru_cache(CacheMax)
    def _from_class(cls, dependent: type[T], *, reuse: Maybe[bool], ignore: NodeIgnore) -> "DependentNode":
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
    def from_node(
        cls,
        factory_or_class: INode[P, T],
        *,
        reuse: Maybe[bool] = MISSING,
        ignore: NodeIgnoreConfig = EmptyIgnore,
    ) -> "DependentNode":
        """
        Build a node Nonrecursively
        from
            - class,
            - async / factory function of the class
        """
        if not is_provided(ignore):
            validated_ignore: NodeIgnore = tuple[Any, ...]()
        elif not isinstance(ignore, tuple):
            validated_ignore: NodeIgnore = (ignore,)
        else:
            validated_ignore = ignore

        if is_class(factory_or_class):
            dependent = cast(type, factory_or_class)
            return cls._from_class(dependent, reuse=reuse,ignore=validated_ignore)
        elif isinstance(factory_or_class, (FunctionType, MethodType)):
            factory = cast(INodeFactory[P, T], factory_or_class)
            return cls._from_function(factory, reuse=reuse, ignore=validated_ignore)
        else:
            raise NotSupportedError(f"Invalid dependent type {factory_or_class}")
