import mobula
from mobula.testing import assert_almost_equal
import numpy as np


@mobula.op.register
class MulOP:
    def forward(self, a, b):
        return a * b

    def backward(self, dy):
        return dy * self.X[1], dy * self.X[0]

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


def test_ctx_mxnet():
    try:
        import mxnet as mx
    except ImportError:
        return
    shape = (5, 5)

    a_np = np.random.random(shape)
    b_np = np.random.random(shape)
    dy_np = np.random.random(shape)

    a = mx.nd.array(a_np)
    b = mx.nd.array(b_np)
    dy = mx.nd.array(dy_np)

    a.attach_grad()
    b.attach_grad()
    with mx.autograd.record():
        c = MulOP(a, b)
    c.backward(dy)
    assert_almost_equal(a.grad, b * dy)
    assert_almost_equal(b.grad, a * dy)
    assert_almost_equal(a * b, c)


def test_ctx_torch():
    try:
        import torch
    except ImportError:
        return
    shape = (5, 5)

    a_np = np.random.random(shape)
    b_np = np.random.random(shape)
    dy_np = np.random.random(shape)

    a = torch.tensor(a_np, requires_grad=True)
    b = torch.tensor(b_np, requires_grad=True)
    dy = torch.tensor(dy_np)
    c = MulOP(a, b)
    c.backward(dy)
    assert_almost_equal(a.grad.data, (b * dy).data)
    assert_almost_equal(b.grad.data, (a * dy).data)
    assert_almost_equal((a * b).data, c.data)


def test_ctx_np():
    shape = (5, 5)
    a = np.random.random(shape)
    b = np.random.random(shape)
    dy = np.random.random(shape)
    op = MulOP[np.ndarray]()
    c = op.forward(a, b)
    a_grad, b_grad = op.backward(dy)
    assert_almost_equal(a_grad, b * dy)
    assert_almost_equal(b_grad, a * dy)
    assert_almost_equal(a * b, c)
