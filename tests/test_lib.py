import mxnet as mx
import numpy as np
import mobula_op
from nose.tools import nottest
from mobula_op.test_utils import assert_almost_equal

def test_lib_add_mx():
    dtype = np.float32
    a = mx.nd.array([1,2,3], dtype = dtype)
    b = mx.nd.array([4,5,6], dtype = dtype)
    c = mx.nd.array([0,0,0], dtype = dtype)
    mobula_op.func.add(a.size, a, b, c)
    assert ((a + b).asnumpy() == c.asnumpy()).all(), c

def test_lib_abs_np():
    dtype = np.float32
    a = np.random.randint(-100, 100, size = (10, 10)).astype(dtype)

    c = np.zeros_like(a, dtype = dtype)
    mobula_op.math.abs(a, out = c)
    assert (np.abs(a) == c).all(), c

    c = mobula_op.math.abs(a)
    assert (np.abs(a) == c).all(), c

def test_binary_op():
    dtype = np.float32
    N, C, H, W = 1, 3, 4, 4
    func = dict(
        add = lambda x, y : x + y,
        sub = lambda x, y : x - y,
        mul = lambda x, y : x * y,
        div = lambda x, y : x / y,
    )
    M = mobula_op.math
    a = np.random.random((N, C, H, W)).astype(dtype)
    b = np.random.random((N, C, H, W)).astype(dtype)
    b[b == 0] = 1.0
    for name, real_f in func.items():
        c = np.empty_like(a, dtype = dtype)
        mf = getattr(M, name)
        mf(a, b, out = c)
        rc = real_f(a, b)
        assert_almost_equal(c, rc)

        d = mf(a, b)
        assert_almost_equal(d, rc)

def test_dot():
    dtype = np.float32
    I, J, U = 3,4,5
    K, M = 6,7
    a = np.random.random((I, J, U)).astype(dtype)
    b = np.random.random((K, U, M)).astype(dtype)
    rc = np.dot(a, b)
    c = mobula_op.math.dot(a, b)
    assert_almost_equal(rc, c)

def test_lib_continuous_mx():
    dtype = np.float32
    shape = (10, 10)
    a = mx.nd.random.uniform(-100, 100, shape, dtype = dtype)
    b = mx.nd.random.uniform(-100, 100, shape, dtype = dtype)
    # a = mx.nd.array(np.random.randint(-100, 100, size = shape), dtype = dtype)
    # b = mx.nd.array(np.random.randint(-100, 100, size = shape), dtype = dtype)
    c = mx.nd.empty((5, 5), dtype = dtype)
    sa = a[2:7, 3:8]
    sb = b[1:6, 4:9]
    # [NOTICE] prepare data
    '''
    NDArray is an asynchronize computation object whose content may not be available.
    [link](https://github.com/apache/incubator-mxnet/issues/2033)
    '''
    mx.nd.waitall()
    # sa and sb are both copies rather than views.
    assert sa.iscontiguous() == True
    assert sb.iscontiguous() == True
    assert c.iscontiguous() == True
    mobula_op.func.add(c.size, sa, sb, c)
    assert (c.asnumpy() == (sa + sb).asnumpy()).all(), (c, (sa + sb))

def test_lib_continuous_np():
    dtype = np.float32
    shape = (10, 10)
    a = np.random.randint(-100, 100, shape).astype(dtype)
    b = np.random.randint(-100, 100, shape).astype(dtype)
    c = np.empty((10, 10), dtype = dtype)
    sa = a[2:7, 3:8]
    sb = b[1:6, 4:9]
    sc = c[3:8, 1:6]
    mobula_op.func.add(sc.size, sa, sb, sc)
    tc = c[3:8, 1:6]
    assert (tc == (sa + sb)).all(), (tc, (sa + sb))

'''
def test_print_carray():
    mobula_op.func.print_carray((1.0, 2.0, 3.0))
'''

if __name__ == '__main__':
    test_lib_add_mx()
    test_lib_add_np()
    test_lib_continuous_mx()
    test_lib_continuous_np()
