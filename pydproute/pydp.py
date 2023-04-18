# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _pydp
else:
    import _pydp

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _pydp.delete_SwigPyIterator

    def value(self):
        return _pydp.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _pydp.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _pydp.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _pydp.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _pydp.SwigPyIterator_equal(self, x)

    def copy(self):
        return _pydp.SwigPyIterator_copy(self)

    def next(self):
        return _pydp.SwigPyIterator_next(self)

    def __next__(self):
        return _pydp.SwigPyIterator___next__(self)

    def previous(self):
        return _pydp.SwigPyIterator_previous(self)

    def advance(self, n):
        return _pydp.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _pydp.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _pydp.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _pydp.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _pydp.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _pydp.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _pydp.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _pydp:
_pydp.SwigPyIterator_swigregister(SwigPyIterator)

class double_array_py(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, nelements):
        _pydp.double_array_py_swiginit(self, _pydp.new_double_array_py(nelements))
    __swig_destroy__ = _pydp.delete_double_array_py

    def __getitem__(self, index):
        return _pydp.double_array_py___getitem__(self, index)

    def __setitem__(self, index, value):
        return _pydp.double_array_py___setitem__(self, index, value)

    def cast(self):
        return _pydp.double_array_py_cast(self)

    @staticmethod
    def frompointer(t):
        return _pydp.double_array_py_frompointer(t)

# Register double_array_py in _pydp:
_pydp.double_array_py_swigregister(double_array_py)

def double_array_py_frompointer(t):
    return _pydp.double_array_py_frompointer(t)

class int_array_py(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self, nelements):
        _pydp.int_array_py_swiginit(self, _pydp.new_int_array_py(nelements))
    __swig_destroy__ = _pydp.delete_int_array_py

    def __getitem__(self, index):
        return _pydp.int_array_py___getitem__(self, index)

    def __setitem__(self, index, value):
        return _pydp.int_array_py___setitem__(self, index, value)

    def cast(self):
        return _pydp.int_array_py_cast(self)

    @staticmethod
    def frompointer(t):
        return _pydp.int_array_py_frompointer(t)

# Register int_array_py in _pydp:
_pydp.int_array_py_swigregister(int_array_py)

def int_array_py_frompointer(t):
    return _pydp.int_array_py_frompointer(t)


def new_doubleP():
    return _pydp.new_doubleP()

def copy_doubleP(value):
    return _pydp.copy_doubleP(value)

def delete_doubleP(obj):
    return _pydp.delete_doubleP(obj)

def doubleP_assign(obj, value):
    return _pydp.doubleP_assign(obj, value)

def doubleP_value(obj):
    return _pydp.doubleP_value(obj)

def new_intP():
    return _pydp.new_intP()

def copy_intP(value):
    return _pydp.copy_intP(value)

def delete_intP(obj):
    return _pydp.delete_intP(obj)

def intP_assign(obj, value):
    return _pydp.intP_assign(obj, value)

def intP_value(obj):
    return _pydp.intP_value(obj)
class DoubleVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _pydp.DoubleVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _pydp.DoubleVector___nonzero__(self)

    def __bool__(self):
        return _pydp.DoubleVector___bool__(self)

    def __len__(self):
        return _pydp.DoubleVector___len__(self)

    def __getslice__(self, i, j):
        return _pydp.DoubleVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _pydp.DoubleVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _pydp.DoubleVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _pydp.DoubleVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _pydp.DoubleVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _pydp.DoubleVector___setitem__(self, *args)

    def pop(self):
        return _pydp.DoubleVector_pop(self)

    def append(self, x):
        return _pydp.DoubleVector_append(self, x)

    def empty(self):
        return _pydp.DoubleVector_empty(self)

    def size(self):
        return _pydp.DoubleVector_size(self)

    def swap(self, v):
        return _pydp.DoubleVector_swap(self, v)

    def begin(self):
        return _pydp.DoubleVector_begin(self)

    def end(self):
        return _pydp.DoubleVector_end(self)

    def rbegin(self):
        return _pydp.DoubleVector_rbegin(self)

    def rend(self):
        return _pydp.DoubleVector_rend(self)

    def clear(self):
        return _pydp.DoubleVector_clear(self)

    def get_allocator(self):
        return _pydp.DoubleVector_get_allocator(self)

    def pop_back(self):
        return _pydp.DoubleVector_pop_back(self)

    def erase(self, *args):
        return _pydp.DoubleVector_erase(self, *args)

    def __init__(self, *args):
        _pydp.DoubleVector_swiginit(self, _pydp.new_DoubleVector(*args))

    def push_back(self, x):
        return _pydp.DoubleVector_push_back(self, x)

    def front(self):
        return _pydp.DoubleVector_front(self)

    def back(self):
        return _pydp.DoubleVector_back(self)

    def assign(self, n, x):
        return _pydp.DoubleVector_assign(self, n, x)

    def resize(self, *args):
        return _pydp.DoubleVector_resize(self, *args)

    def insert(self, *args):
        return _pydp.DoubleVector_insert(self, *args)

    def reserve(self, n):
        return _pydp.DoubleVector_reserve(self, n)

    def capacity(self):
        return _pydp.DoubleVector_capacity(self)
    __swig_destroy__ = _pydp.delete_DoubleVector

# Register DoubleVector in _pydp:
_pydp.DoubleVector_swigregister(DoubleVector)

class IntVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _pydp.IntVector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _pydp.IntVector___nonzero__(self)

    def __bool__(self):
        return _pydp.IntVector___bool__(self)

    def __len__(self):
        return _pydp.IntVector___len__(self)

    def __getslice__(self, i, j):
        return _pydp.IntVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _pydp.IntVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _pydp.IntVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _pydp.IntVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _pydp.IntVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _pydp.IntVector___setitem__(self, *args)

    def pop(self):
        return _pydp.IntVector_pop(self)

    def append(self, x):
        return _pydp.IntVector_append(self, x)

    def empty(self):
        return _pydp.IntVector_empty(self)

    def size(self):
        return _pydp.IntVector_size(self)

    def swap(self, v):
        return _pydp.IntVector_swap(self, v)

    def begin(self):
        return _pydp.IntVector_begin(self)

    def end(self):
        return _pydp.IntVector_end(self)

    def rbegin(self):
        return _pydp.IntVector_rbegin(self)

    def rend(self):
        return _pydp.IntVector_rend(self)

    def clear(self):
        return _pydp.IntVector_clear(self)

    def get_allocator(self):
        return _pydp.IntVector_get_allocator(self)

    def pop_back(self):
        return _pydp.IntVector_pop_back(self)

    def erase(self, *args):
        return _pydp.IntVector_erase(self, *args)

    def __init__(self, *args):
        _pydp.IntVector_swiginit(self, _pydp.new_IntVector(*args))

    def push_back(self, x):
        return _pydp.IntVector_push_back(self, x)

    def front(self):
        return _pydp.IntVector_front(self)

    def back(self):
        return _pydp.IntVector_back(self)

    def assign(self, n, x):
        return _pydp.IntVector_assign(self, n, x)

    def resize(self, *args):
        return _pydp.IntVector_resize(self, *args)

    def insert(self, *args):
        return _pydp.IntVector_insert(self, *args)

    def reserve(self, n):
        return _pydp.IntVector_reserve(self, n)

    def capacity(self):
        return _pydp.IntVector_capacity(self)
    __swig_destroy__ = _pydp.delete_IntVector

# Register IntVector in _pydp:
_pydp.IntVector_swigregister(IntVector)


def run_dp(n, m, f, D, I, J, V, c, T, S, a, b, C, verbose, inexact, timelimit):
    return _pydp.run_dp(n, m, f, D, I, J, V, c, T, S, a, b, C, verbose, inexact, timelimit)

