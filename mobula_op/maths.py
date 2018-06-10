import functools
from . import func
from . import glue

def unary_op(f, a, **kwargs):
    # [TODO] broadcast
    out = kwargs.get('out', None)
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a)
    f(a.size, a, out)
    return out

def binary_op(f, a, b, **kwargs):
    assert type(a) == type(b)
    assert a.shape == b.shape
    # [TODO] broadcast
    out = kwargs.get('out', None)
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a)
    f(a.size, a, b, out)
    return out

abs = functools.partial(unary_op, func.abs)

add = functools.partial(binary_op, func.add)
sub = functools.partial(binary_op, func.sub)
mul = functools.partial(binary_op, func.mul)
div = functools.partial(binary_op, func.div)
