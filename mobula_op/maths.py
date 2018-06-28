import functools
import operator
from . import func
from . import glue
from . import const

def is_same_shape(a, b):
    return tuple(a) == tuple(b)

def unary_op(f, a, out = None):
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a, dtype = a.dtype)
    else:
        assert is_same_shape(out.shape, a.shape)
    f(a.size, a, out)
    return out

def binary_op(f, a, b, out = None):
    assert type(a) == type(b)
    assert is_same_shape(a.shape, b.shape)
    # [TODO] broadcast
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty_like(a, dtype = a.dtype)
    else:
        assert is_same_shape(out.shape, a.shape)
    f(a.size, a, b, out)
    return out

abs = functools.partial(unary_op, func.abs)

add = functools.partial(binary_op, func.add)
sub = functools.partial(binary_op, func.sub)
mul = functools.partial(binary_op, func.mul)
div = functools.partial(binary_op, func.div)

def dot(a, b, out = None, req = const.req.write):
    '''
    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    numpy.dot and mxnet.nd.dot are different.
    numpy.dot: dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    mxnet.nd.dot: dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
    '''
    assert a.ndim >= 2 and b.ndim >= 2
    U = a.shape[-1]
    assert b.shape[-2] == U
    bshape = b.shape
    M = bshape[-1]
    I = a.size / U
    K = b.size / (bshape[-1] * U)
    out_shape = a.shape[:-1] + b.shape[:-2] + (b.shape[-1], )
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.zeros(out_shape, dtype = a.dtype)
    else:
        assert is_same_shape(out.shape, out_shape)
        if req != const.req.add:
            out[:] = 0
    func.dot_add(a, b, I, U, K, M, out)
    return out

def transpose(data, axes, out = None):
    assert data.ndim == len(axes)
    vis = [False for _ in range(data.ndim)]
    for a in axes:
        assert a >= 0
        assert vis[a] == False
        vis[a] = True
    out_shape = [data.shape[i] for i in axes]
    if out is None:
        backend = glue.backend.get_var_backend(data)
        out = backend.F.empty(out_shape, dtype = data.dtype)
    else:
        assert out.shape == out_shape
    func.transpose(data, data.shape, axes, out)
    return out

def tensordot(a, b, axes = 2, out = None):
    if isinstance(axes, int):
        axes = ([-axes], [axes])
    assert len(axes) == 2
    assert len(axes[0]) == len(axes[1])
    def get_pos_axes(axes, ndim):
        res = []
        for x in axes:
            if x < 0:
                x += ndim
                assert x >= 0
            res.append(x)
        return res

    a_axes = get_pos_axes(axes[0], a.ndim)
    b_axes = get_pos_axes(axes[1], b.ndim)

    def get_rest_axes(axes, ndim):
        return [i for i in range(ndim) if i not in axes]

    a_rest_axes = get_rest_axes(a_axes, a.ndim)
    b_rest_axes = get_rest_axes(b_axes, b.ndim)

    def get_shape_i(axes, shape):
        return [shape[x] for x in axes]

    a_axes_shape = get_shape_i(a_axes, a.shape)
    b_axes_shape = get_shape_i(b_axes, b.shape)

    for ax, bx in zip(a_axes_shape, b_axes_shape):
        assert ax == bx

    a_rest_axes_shape = get_shape_i(a_rest_axes, a.shape)
    b_rest_axes_shape = get_shape_i(b_rest_axes, b.shape)

    prod = lambda x : functools.reduce(operator.mul, x)

    a_transpose = transpose(a, a_rest_axes + a_axes).reshape((prod(a_rest_axes_shape), -1))
    b_transpose = transpose(b, b_axes + b_rest_axes).reshape((-1, prod(b_rest_axes_shape)))

    out_shape = a_rest_axes_shape + b_rest_axes_shape
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.empty(out_shape, dtype = a.dtype)
    else:
        assert is_same_shape(out.shape, out_shape)

    dot(a_transpose, b_transpose, out = out.reshape(a_transpose.shape[0], b_transpose.shape[1]))
    return out

LINALG_GEMM_FUNC = [
    [func.linalg_gemm_ff, func.linalg_gemm_ft],
    [func.linalg_gemm_tf, func.linalg_gemm_tt]
]

def linalg_gemm(a, b, out = None, tA = False, tB = False, req = const.req.write):
    assert a.ndim == 2 and b.ndim == 2
    a_shape = a.shape[::-1] if tA else a.shape
    b_shape = b.shape[::-1] if tB else b.shape
    assert a_shape[-1] == b_shape[0]
    I, U = a_shape
    J = b_shape[-1]
    out_shape = (I, J)
    if out is None:
        backend = glue.backend.get_var_backend(a)
        out = backend.F.zeros(out_shape, dtype = a.dtype)
    else:
        assert is_same_shape(out.shape, out_shape)
        if req != const.req.add:
            out[:] = 0
    LINALG_GEMM_FUNC[tA][tB](a, b, I, U, J, out)
