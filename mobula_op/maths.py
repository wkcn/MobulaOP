import functools
import operator
from . import func
from . import glue

def unary_op(f, a, **kwargs):
    out = kwargs.get('out', None)
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a, dtype = a.dtype)
    else:
        assert out.shape == a.shape
    f(a.size, a, out)
    return out

def binary_op(f, a, b, **kwargs):
    assert type(a) == type(b)
    assert a.shape == b.shape
    # [TODO] broadcast
    out = kwargs.get('out', None)
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a, dtype = a.dtype)
    else:
        assert out.shape == a.shape
    f(a.size, a, b, out)
    return out

abs = functools.partial(unary_op, func.abs)

add = functools.partial(binary_op, func.add)
sub = functools.partial(binary_op, func.sub)
mul = functools.partial(binary_op, func.mul)
div = functools.partial(binary_op, func.div)

def dot(a, b, **kwargs):
    U = a.shape[-1]
    assert b.shape[-2] == U
    bshape = b.shape
    M = bshape[-1]
    I = a.size / U
    K = b.size / (bshape[-1] * U)
    out = kwargs.get('out', None)
    out_shape = a.shape[:-1] + b.shape[:-2] + (b.shape[-1], )
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty(out_shape, dtype = a.dtype)
    else:
        assert out.shape == out_shape
    func.dot(a, b, I, U, K, M, out)
    return out
