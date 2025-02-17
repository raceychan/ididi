/_node.c:1525:13: error: expected identifier or ‘(’ before ‘default’
 1525 |   PyObject *default;
      |             ^~~~~~~
In file included from .pixi/envs/default/include/python3.9/pytime.h:6,
                 from .pixi/envs/default/include/python3.9/Python.h:85,
                 from /_node.c:28:
/_node.c: In function ‘__pyx_pf_5ididi_5_node_10Dependency___init__’:
/_node.c:5914:30: error: expected identifier before ‘default’
 5914 |   __Pyx_DECREF(__pyx_v_self->default);
      |                              ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:112:41: note: in definition of macro ‘_PyObject_CAST’
  112 | #define _PyObject_CAST(op) ((PyObject*)(op))
      |                                         ^~
/_node.c:1625:27: note: in expansion of macro ‘Py_DECREF’
 1625 |   #define __Pyx_DECREF(r) Py_DECREF(r)
      |                           ^~~~~~~~~
/_node.c:5914:3: note: in expansion of macro ‘__Pyx_DECREF’
 5914 |   __Pyx_DECREF(__pyx_v_self->default);
      |   ^~~~~~~~~~~~
/_node.c:5915:17: error: expected identifier before ‘default’
 5915 |   __pyx_v_self->default = __pyx_v_default;
      |                 ^~~~~~~
/_node.c: In function ‘__pyx_pf_5ididi_5_node_10Dependency_2__repr__’:
/_node.c:6051:80: error: expected identifier before ‘default’
 6051 |   __pyx_t_5 = __Pyx_PyObject_FormatSimpleAndDecref(PyObject_Repr(__pyx_v_self->default), __pyx_empty_unicode); if (unlikely(!__pyx_t_5)) __PYX_ERR(0, 182, __pyx_L1_error)
      |                                                                                ^~~~~~~
/_node.c: In function ‘__pyx_pf_5ididi_5_node_10Dependency_4replace_type’:
/_node.c:6315:66: error: expected identifier before ‘default’
 6315 |   if (PyDict_SetItem(__pyx_t_1, __pyx_n_s_default, __pyx_v_self->default) < 0) __PYX_ERR(0, 190, __pyx_L1_error)
      |                                                                  ^~~~~~~
In file included from .pixi/envs/default/include/python3.9/pytime.h:6,
                 from .pixi/envs/default/include/python3.9/Python.h:85,
                 from /_node.c:28:
/_node.c: In function ‘__pyx_pf_5ididi_5_node_10Dependency_7default___get__’:
/_node.c:6455:30: error: expected identifier before ‘default’
 6455 |   __Pyx_INCREF(__pyx_v_self->default);
      |                              ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:112:41: note: in definition of macro ‘_PyObject_CAST’
  112 | #define _PyObject_CAST(op) ((PyObject*)(op))
      |                                         ^~
/_node.c:1624:27: note: in expansion of macro ‘Py_INCREF’
 1624 |   #define __Pyx_INCREF(r) Py_INCREF(r)
      |                           ^~~~~~~~~
/_node.c:6455:3: note: in expansion of macro ‘__Pyx_INCREF’
 6455 |   __Pyx_INCREF(__pyx_v_self->default);
      |   ^~~~~~~~~~~~
/_node.c:6456:27: error: expected identifier before ‘default’
 6456 |   __pyx_r = __pyx_v_self->default;
      |                           ^~~~~~~
In file included from .pixi/envs/default/include/python3.9/pytime.h:6,
                 from .pixi/envs/default/include/python3.9/Python.h:85,
                 from /_node.c:28:
/_node.c: In function ‘__pyx_pf_5ididi_5_node_10Dependency_6__reduce_cython__’:
/_node.c:6587:30: error: expected identifier before ‘default’
 6587 |   __Pyx_INCREF(__pyx_v_self->default);
      |                              ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:112:41: note: in definition of macro ‘_PyObject_CAST’
  112 | #define _PyObject_CAST(op) ((PyObject*)(op))
      |                                         ^~
/_node.c:1624:27: note: in expansion of macro ‘Py_INCREF’
 1624 |   #define __Pyx_INCREF(r) Py_INCREF(r)
      |                           ^~~~~~~~~
/_node.c:6587:3: note: in expansion of macro ‘__Pyx_INCREF’
 6587 |   __Pyx_INCREF(__pyx_v_self->default);
      |   ^~~~~~~~~~~~
In file included from .pixi/envs/default/include/python3.9/tupleobject.h:39,
                 from .pixi/envs/default/include/python3.9/Python.h:105,
                 from /_node.c:28:
/_node.c:6589:58: error: expected identifier before ‘default’
 6589 |   if (__Pyx_PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_v_self->default)) __PYX_ERR(2, 5, __pyx_L1_error);
      |                                                          ^~~~~~~
.pixi/envs/default/include/python3.9/cpython/tupleobject.h:30:69: note: in definition of macro ‘PyTuple_SET_ITEM’
   30 | #define PyTuple_SET_ITEM(op, i, v) (_PyTuple_CAST(op)->ob_item[i] = v)
      |                                                                     ^
/_node.c:6589:7: note: in expansion of macro ‘__Pyx_PyTuple_SET_ITEM’
 6589 |   if (__Pyx_PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_v_self->default)) __PYX_ERR(2, 5, __pyx_L1_error);
      |       ^~~~~~~~~~~~~~~~~~~~~~
/_node.c:1105:69: warning: left-hand operand of comma expression has no effect [-Wunused-value]
 1105 |   #define __Pyx_PyTuple_SET_ITEM(o, i, v) (PyTuple_SET_ITEM(o, i, v), (0))
      |                                                                     ^
/_node.c:6589:7: note: in expansion of macro ‘__Pyx_PyTuple_SET_ITEM’
 6589 |   if (__Pyx_PyTuple_SET_ITEM(__pyx_t_2, 0, __pyx_v_self->default)) __PYX_ERR(2, 5, __pyx_L1_error);
      |       ^~~~~~~~~~~~~~~~~~~~~~
/_node.c:6669:32: error: expected identifier before ‘default’
 6669 |     __pyx_t_4 = (__pyx_v_self->default != Py_None);
      |                                ^~~~~~~
In file included from .pixi/envs/default/include/python3.9/pytime.h:6,
                 from .pixi/envs/default/include/python3.9/Python.h:85,
                 from /_node.c:28:
/_node.c: In function ‘__pyx_f_5ididi_5_node___pyx_unpickle_Dependency__set_state’:
/_node.c:13235:38: error: expected identifier before ‘default’
13235 |   __Pyx_DECREF(__pyx_v___pyx_result->default);
      |                                      ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:112:41: note: in definition of macro ‘_PyObject_CAST’
  112 | #define _PyObject_CAST(op) ((PyObject*)(op))
      |                                         ^~
/_node.c:1625:27: note: in expansion of macro ‘Py_DECREF’
 1625 |   #define __Pyx_DECREF(r) Py_DECREF(r)
      |                           ^~~~~~~~~
/_node.c:13235:3: note: in expansion of macro ‘__Pyx_DECREF’
13235 |   __Pyx_DECREF(__pyx_v___pyx_result->default);
      |   ^~~~~~~~~~~~
/_node.c:13236:25: error: expected identifier before ‘default’
13236 |   __pyx_v___pyx_result->default = __pyx_t_1;
      |                         ^~~~~~~
/_node.c: In function ‘__pyx_tp_new_5ididi_5_node_Dependency’:
/_node.c:13383:6: error: expected identifier before ‘default’
13383 |   p->default = Py_None; Py_INCREF(Py_None);
      |      ^~~~~~~
In file included from .pixi/envs/default/include/python3.9/pytime.h:6,
                 from .pixi/envs/default/include/python3.9/Python.h:85,
                 from /_node.c:28:
/_node.c: In function ‘__pyx_tp_dealloc_5ididi_5_node_Dependency’:
/_node.c:13399:15: error: expected identifier before ‘default’
13399 |   Py_CLEAR(p->default);
      |               ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:112:41: note: in definition of macro ‘_PyObject_CAST’
  112 | #define _PyObject_CAST(op) ((PyObject*)(op))
      |                                         ^~
/_node.c:13399:3: note: in expansion of macro ‘Py_CLEAR’
13399 |   Py_CLEAR(p->default);
      |   ^~~~~~~~
/_node.c:13399:15: error: expected identifier before ‘default’
13399 |   Py_CLEAR(p->default);
      |               ^~~~~~~
.pixi/envs/default/include/python3.9/object.h:479:14: note: in definition of macro ‘Py_CLEAR’
  479 |             (op) = NULL;                        \
      |              ^~
/_node.c: In function ‘__pyx_tp_traverse_5ididi_5_node_Dependency’:
/_node.c:13416:10: error: expected identifier before ‘default’
13416 |   if (p->default) {
      |          ^~~~~~~
/_node.c:13417:17: error: expected identifier before ‘default’
13417 |     e = (*v)(p->default, a); if (e) return e;
      |                 ^~~~~~~
/_node.c:13417:10: error: too few arguments to function ‘v’
13417 |     e = (*v)(p->default, a); if (e) return e;
      |         ~^~~
/_node.c: In function ‘__pyx_tp_clear_5ididi_5_node_Dependency’:
/_node.c:13428:24: error: expected identifier before ‘default’
13428 |   tmp = ((PyObject*)p->default);
      |                        ^~~~~~~
/_node.c:13429:6: error: expected identifier before ‘default’
13429 |   p->default = Py_None; Py_INCREF(Py_None);