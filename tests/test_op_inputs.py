import mobula_op
import mxnet as mx
import numpy as np
from mobula_op.test_utils import assert_almost_equal

try:
    import torch
except ImportError:
    torch = None

@mobula_op.operator.register
class TestInputsOP:
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

def check_op_inputs_torch(test_op):
    a = torch.tensor([1.0,2.0,3.0], requires_grad = True)
    b = torch.tensor([4.0,5.0,6.0], requires_grad = True)
    c = test_op(a, b)
    dy = torch.tensor([10.0,11.0,12.0])
    c.backward(dy)
    assert (a.grad == (b * dy)).all(), a.grad
    assert (b.grad == (a * dy)).all(), b.grad
    assert ((a * b) == c).all()

def test_op_inputs():
    for op in [TestInputsOP, TestInputsOP2]:
        check_op_inputs(op)
        check_op_inputs_np(op)
        if torch is not None:
            check_op_inputs_torch(op)

@mobula_op.operator.register
class AttrOP:
    def __init__(self, value):
        self.value = value
    def forward(self, x):
        self.y[:] = x + self.value
    def backward(self, dy):
        self.dx[:] = dy
    def infer_shape(self, in_shape):
        return in_shape, in_shape

def test_attr_op():
    dtype = np.float32
    shape = (1, )
    x_sym = mx.sym.Variable('x')
    y_sym = AttrOP(x_sym, value = 3) + AttrOP(x_sym, value = 4)
    exe = y_sym.simple_bind(ctx = mx.context.current_context(), x = shape)
    exe.forward(x = mx.nd.array([0]))
    res = exe.outputs[0]
    gt = 3 + 4 + 0
    assert_almost_equal(res.asscalar(), gt)

if __name__ == '__main__':
    test_op_inputs()
