from types import MappingProxyType

from .node import DependentNode

type NodeDependent[T] = type[T]
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

type ResolvedInstances[T] = dict[type[T], T]
"""
mapping a type to its resolved instance
"""

type TypeMappings[T] = dict[NodeDependent[T], list[NodeDependent[T]]]
"""
### mapping a type to its dependencies
"""
