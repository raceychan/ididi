from collections import defaultdict
from types import MappingProxyType

from .node import DependentNode, ForwardDependent

type Dependent[I] = type[I] | ForwardDependent[I]
"""
### A dependent can be a concrete type or a forward reference
"""

type GraphNodes[I] = dict[Dependent[I], DependentNode[I]]
"""
### mapping a type to its corresponding node
"""
type GraphNodesView[I] = MappingProxyType[Dependent[I], DependentNode[I]]
"""
### a readonly view of GraphNodes
"""

type ResolvedInstances[I] = dict[Dependent[I], I]
"""
mapping a type to its resolved instance
"""

type TypeMappings[I] = dict[type[I], list[type[I]]]
"""
### mapping a type to its dependencies
"""

type TypeMappingView[I] = MappingProxyType[type[I], list[type[I]]]
"""
### a readonly view of TypeMappings
"""

type Neighbors[I] = defaultdict[Dependent[I], list[Dependent[I]]]
"""
### mapping a dependent to its neighbors, could be used to represent its dependents or its dependencies
"""

type NeighborsView[I] = MappingProxyType[type[I], list[Dependent[I]]]
"""
### a readonly view of Neighbors
"""
