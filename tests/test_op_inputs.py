import mobula_op
import mxnet as mx
import numpy as np

@mobula_op.operator.register
class TestInputsOP:
    def __init__(self):
        pass
    def forward(self, x, y):
        self.y[:] = self.X[0] * self.X[1]
    def backward(self, dy): 
        self.dX[0][:] = dy * self.X[1]
        self.assign(self.dX[1], self.req[1], dy * self.X[0])
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

def test_op_inputs():
    a = mx.nd.array([1,2,3]) 
    b = mx.nd.array([4,5,6])
    a.attach_grad()
    b.attach_grad()
    with mx.autograd.record():
        c = TestInputsOP(a, b)
    dy = mx.nd.array([10,11,12])
    c.backward(dy)
    assert (a.grad.asnumpy() == (b * dy).asnumpy()).all(), a.grad
    assert (b.grad.asnumpy() == (a * dy).asnumpy()).all(), b.grad
    assert ((a * b).asnumpy() == c.asnumpy()).all()

if __name__ == '__main__':
    test_op_inputs()
