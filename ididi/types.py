# import typing as ty
from types import MappingProxyType

from .node import AbstractDependent, DependentNode, ForwardDependent

type NodeDependent[I] = type[I] | ForwardDependent[I]
"""
### A dependent can be a concrete type or a forward reference
"""

type GraphNodes[I: type] = dict[NodeDependent[I], DependentNode[I]]
"""
### mapping a type to its corresponding node
"""

type GraphNodesView[I: type] = MappingProxyType[NodeDependent[I], DependentNode[I]]
"""
### a readonly view of GraphNodes
"""

type ResolvedInstances[I] = dict[NodeDependent[I], I]
"""
mapping a type to its resolved instance
"""

type TypeMappings[I] = dict[NodeDependent[I], list[NodeDependent[I]]]
"""
### mapping a type to its dependencies
"""

type TypeMappingView[I: type] = MappingProxyType[
    NodeDependent[I], list[NodeDependent[I]]
]
"""
### a readonly view of TypeMappings
"""

# type Neighbors[I] = defaultdict[Dependent[I], list[Dependent[I]]]
# """
# ### mapping a dependent to its neighbors, could be used to represent its dependents or its dependencies
# """

# type NeighborsView[I] = MappingProxyType[type[I], list[Dependent[I]]]
# """
# ### a readonly view of Neighbors
# """
