import mxnet as mx
import numpy as np
import mobula_op
from nose.tools import nottest

def test_lib_add_mx():
    dtype = np.float32
    a = mx.nd.array([1,2,3], dtype = dtype)
    b = mx.nd.array([4,5,6], dtype = dtype)
    c = mx.nd.array([0,0,0], dtype = dtype)
    mobula_op.func.add(a.size, a, b, c)
    assert ((a + b).asnumpy() == c.asnumpy()).all(), c

def test_lib_add_np():
    dtype = np.float32
    a = np.array([1,2,3], dtype = dtype)
    b = np.array([4,9,11], dtype = dtype)
    c = np.array([0,0,0], dtype = dtype)
    mobula_op.func.add(a.size, a, b, c)
    assert ((a + b) == c).all(), c

def test_lib_sub_np():
    dtype = np.float32
    a = np.array([1,2,3], dtype = dtype)
    b = np.array([9,54,32], dtype = dtype)
    c = np.array([0,0,0], dtype = dtype)
    mobula_op.func.sub(a.size, a, b, c)
    assert ((a - b) == c).all(), c

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

if __name__ == '__main__':
    test_lib_add_mx()
    test_lib_add_np()
    test_lib_continuous_mx()
    test_lib_continuous_np()
