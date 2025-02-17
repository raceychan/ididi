cdef class Dependency:
    cdef readonly str name
    cdef readonly object param_type
    cdef readonly object default_
    cdef readonly bint unresolvable

cdef class Dependencies(dict):
    pass

cdef class DependentNode:
    cdef readonly object dependent
    cdef readonly object factory
    cdef readonly str factory_type
    cdef readonly bint function_dependent
    cdef readonly Dependencies dependencies
    cdef readonly object config