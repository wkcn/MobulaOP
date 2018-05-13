import mxnet as mx
import numpy as np
import mobula

def test_lib():

    dtype = np.float32
    a = mx.nd.array([1,2,3], dtype = dtype)
    b = mx.nd.array([4,5,6], dtype = dtype)
    c = mx.nd.array([0,0,0], dtype = dtype)

    print (mobula.func)
    mobula.func.add(a.size, a, b, c)

    assert ((a + b).asnumpy() == c.asnumpy()).all(), c

if __name__ == '__main__':
    test_lib()
