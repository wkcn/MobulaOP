import mxnet as mx
import numpy as np
import mobula

def test_lib():
    mobula_op = mobula.load_lib('./mobula/build/mobula_op_cpu.so')

    dtype = np.float32
    a = mx.nd.array([1,2,3], dtype = dtype)
    b = mx.nd.array([4,5,6], dtype = dtype)
    c = mx.nd.array([0,0,0], dtype = dtype)

    mobula_op.add(a.size, a, b, c)

    assert ((a + b).asnumpy() == c.asnumpy()).all(), c
