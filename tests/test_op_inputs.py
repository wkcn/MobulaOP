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

@mobula_op.operator.register
class TestInputsOP2:
    def __init__(self):
        pass
    def forward(self, x, y):
        return self.X[0] * self.X[1]
    def backward(self, dy):
        return dy * self.X[1], dy * self.X[0]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

def check_op_inputs(test_op):
    a = mx.nd.array([1,2,3]) 
    b = mx.nd.array([4,5,6])
    a.attach_grad()
    b.attach_grad()
    with mx.autograd.record():
        c = test_op(a, b)
    dy = mx.nd.array([10,11,12])
    c.backward(dy)
    assert (a.grad.asnumpy() == (b * dy).asnumpy()).all(), a.grad
    assert (b.grad.asnumpy() == (a * dy).asnumpy()).all(), b.grad
    assert ((a * b).asnumpy() == c.asnumpy()).all()

def check_op_inputs_np(test_op):
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    op = test_op[np.ndarray]()

    c = op(a, b)
    assert ((a * b) == c).all()

    c = op.forward(a, b)
    assert ((a * b) == c).all()

    dy = np.array([10,11,12])
    dX1 = op.backward(out_grad = dy)
    dX2 = op.backward(dy)
    for x1, x2 in zip(dX1, dX2):
        assert (x1 == x2).all()
    a_grad, b_grad = dX1
    assert (a_grad == (b * dy)).all(), a_grad
    assert (b_grad == (a * dy)).all(), b_grad

def test_op_inputs():
    for op in [TestInputsOP, TestInputsOP2]:
        check_op_inputs(op)
        check_op_inputs_np(op)

if __name__ == '__main__':
    test_op_inputs()
