import mobula_op
import mxnet as mx
import numpy as np
from mobula_op.test_utils import assert_almost_equal

@mobula_op.operator.register('FirstOP')
class FirstOP:
    def __init__(self, par):
        assert type(par) == dict
        self.par = par
        assert self.par == {'a': 3} 
    def forward(self, x, y):
        return x + y
    def backward(self, dy): 
        return [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

def test_register_op():
    a = mx.nd.array([1,2,3]) 
    b = mx.nd.array([4,5,6])
    c1 = FirstOP(a, b, dict(a = 3))
    c2 = FirstOP(a, b, par = dict(a = 3))
    assert_almost_equal(c1.asnumpy(), c2.asnumpy())
    assert_almost_equal((a + b).asnumpy(), c1.asnumpy())
