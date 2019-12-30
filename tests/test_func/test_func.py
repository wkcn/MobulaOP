import mobula
from mobula.testing import assert_almost_equal
import numpy as np
import os
import ctypes
mobula.op.load('./utils', os.path.dirname(__file__))


def test_non_c_contiguous():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    c = np.empty((5, 5))
    s = (slice(None), slice(2, 4))
    a_part = a[s]
    b_part = b[s]
    c_part = c[s]
    assert a_part.flags.c_contiguous == False
    assert b_part.flags.c_contiguous == False
    assert c_part.flags.c_contiguous == False
    mobula.func.mul_elemwise(a_part.size, a_part, b_part, c_part)
    assert_almost_equal(a_part * b_part, c_part)
    assert_almost_equal(c[s], c_part)


def test_func_kwargs():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    c = np.empty((5, 5))
    mobula.func.mul_elemwise(n=a.size, a=a, b=b, c=c)
    assert_almost_equal(a * b, c)


def test_default_value_op():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    value = np.random.random((5, 5))
    op = mobula.op.default_add_op[np.ndarray](value=value)
    c = op(a, b)
    assert_almost_equal(a + b, c)
    c = op(a)  # a+b[default=value]
    assert_almost_equal(a + value, c)


def test_thread():
    n = 300
    out_1 = np.empty(n // 1)
    out_2 = np.empty(n // 2)
    out_3 = np.empty(n // 3)
    out_4 = np.empty(n * 2)
    out_5 = np.empty(n * 3)
    mobula.func.test_thread(n, out_1, out_2, out_3, out_4, out_5)
    assert_almost_equal(np.arange(n // 1) * 1, out_1)
    assert_almost_equal(np.arange(n // 2) * 2, out_2)
    assert_almost_equal(np.arange(n // 3) * 3, out_3)
    assert_almost_equal(np.arange(n * 2) * 2, out_4)
    assert_almost_equal(np.arange(n * 3) * 3, out_5)


def test_const_template():
    shape = (5, 5)
    value = 3939
    cs = [ctypes.c_int, ctypes.c_float, ctypes.c_double]
    vs = [3, 9.9, 3.9]
    atols = [0, 1e-6, 1e-6]
    for ctype, value, atol in zip(cs, vs, atols):
        c_value = ctype(value)
        a = np.empty(shape)
        mobula.func.test_const_template(a.size, c_value, a)
        assert_almost_equal(np.tile(value, shape), a, atol=atol)


def test_infer_type_for_const():
    ns = [np.int32, np.int64, np.float32, np.float64]
    N = 3
    V = 39.39
    for dtype in ns:
        out = np.empty(N, dtype=dtype)
        rv = dtype(V).tolist()
        mobula.func.infer_type_for_const(N, rv, out)
        assert_almost_equal(out, rv)


def test_void_pointer():
    pv = 3939
    p = ctypes.c_void_p(pv)
    out = np.zeros(1, dtype=np.int64)
    mobula.func.test_void_pointer(1, p, out)
    assert out == pv


def test_mobula_func():
    # skip float temporarily
    ns = [np.int32, np.int64]  # , np.float32, np.float64]
    pv = 39.39
    for dtype in ns:
        a = np.array([pv], dtype=dtype)
        b = np.empty(a.shape, dtype=dtype)
        rtn = mobula.func.set_and_return(a, b)
        assert_almost_equal(a, b)
        assert_almost_equal(a, rtn)


def test_build():
    mobula.func.mul_elemwise.build('cpu', ['float'])
    mobula.func.mul_elemwise.build('cpu', dict(T='int'))
    code_fname = os.path.join(os.path.dirname(
        __file__), 'utils/build/cpu/utils_wrapper.cpp')
    code = open(code_fname).read()
    '''
    In windows, `ctypes.c_int` is the same as `ctypes.c_long`, whose name is `c_long`. The function of `get_ctype_name` will return `int32_t` :(
    '''
    assert 'mul_elemwise_kernel<float>' in code, code
    assert 'mul_elemwise_kernel<int' in code, code


def test_atomic_add():
    I = U = J = 100
    import time
    for _ in range(10):
        tic = time.time()
        dtype = np.float32
        a = np.random.uniform(size=(I, U)).astype(dtype).round(1)
        b = np.random.uniform(size=(U, J)).astype(dtype).round(1)
        out = np.zeros((I, J), dtype=dtype)
        mobula.func.test_atomic_add_by_gemm(U, I, J, a, b, out)
        target = np.dot(a, b)
        assert_almost_equal(out, target, atol=1e-3)
        # print('test_atomic_add', time.time() - tic)
