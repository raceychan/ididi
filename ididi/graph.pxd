cdef object _resolve_dfs(Resolver resolver, dict nodes, dict cache, object ptype, dict overrides)


cdef class Resolver:
    cdef readonly dict _nodes
    cdef readonly dict _analyzed_nodes
    cdef readonly object _type_registry
    cdef readonly frozenset _ignore
    cdef readonly object _workers

    cdef readonly set _registered_singletons
    cdef readonly dict _resolved_singletons

cdef class ResolveScope(Resolver):
    cdef readonly str _name
    cdef readonly object _pre
    cdef readonly object _stack


cdef class SyncScope(ResolveScope):
    pass

cdef class AsyncScope(ResolveScope):
    cdef readonly object _loop


cdef class Graph(Resolver):
    cdef object _token