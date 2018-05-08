import mobula
import mxnet as mx
import numpy as np

@mobula.register_op
class TestInputsOP:
    def __init__(self):
        pass
    def forward(self, x, y):
        self.Y[:] = self.X[0] + self.X[1]
    def backward(self, dy): 
        self.dX[:] = [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

def test_op_inputs():
    a = mx.nd.array([1,2,3]) 
    b = mx.nd.array([4,5,6])
    c = TestInputsOP(a, b)
    assert ((a+b).asnumpy() == c.asnumpy()).all()

if __name__ == '__main__':
    test_op_inputs()
