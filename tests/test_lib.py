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
    b = np.array([4,5,6], dtype = dtype)
    c = np.array([0,0,0], dtype = dtype)
    mobula_op.func.add(a.size, a, b, c)
    assert ((a + b) == c).all(), c

# mxnet.ndarray.NDArray doesn't have iscontiguous property :-(
@nottest
def test_lib_continuous_mx():
    dtype = np.float32
    shape = (10, 10)
    a = mx.nd.random.uniform(-100, 100, shape, dtype = dtype)
    b = mx.nd.random.uniform(-100, 100, shape, dtype = dtype)
    c = mx.nd.empty((5,5), dtype = dtype)
    sa = a[2:7, 3:8]
    sb = b[1:6, 4:9]
    mobula_op.func.add(c.size, sa, sb, c)
    assert (c.asnumpy() == (sa + sb).asnumpy()).all(), (c, (sa + sb))

def test_lib_continuous_np():
    dtype = np.float32
    shape = (10, 10)
    a = np.random.randint(-100, 100, shape).astype(dtype)
    b = np.random.randint(-100, 100, shape).astype(dtype)
    c = np.empty((5,5), dtype = dtype)
    sa = a[2:7, 3:8]
    sb = b[1:6, 4:9]
    mobula_op.func.add(c.size, sa, sb, c)
    assert (c == (sa + sb)).all(), (c, (sa + sb))

if __name__ == '__main__':
    test_lib_add_mx()
    test_lib_add_np()
    test_lib_continuous_mx()
    test_lib_continuous_np()
