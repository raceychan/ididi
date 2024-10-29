import typing as ty
from types import MappingProxyType

from .node import DependentNode, ForwardDependent

type NodeDependent = type | ForwardDependent
"""
### A dependent can be a concrete type or a forward reference
"""

type GraphNodes[I] = dict[type[I], DependentNode[I]]
"""
### mapping a type to its corresponding node
"""

type GraphNodesView[I] = MappingProxyType[type[I], DependentNode[I]]
"""
### a readonly view of GraphNodes
"""

type ResolvedInstances = dict[type, ty.Any]
"""
mapping a type to its resolved instance
"""

type TypeMappings = dict[NodeDependent, list[NodeDependent]]
"""
### mapping a type to its dependencies
"""

type TypeMappingView = MappingProxyType[NodeDependent, list[NodeDependent]]
"""
### a readonly view of TypeMappings
"""
