import time
import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal, gradcheck
import unittest

mobula.op.load('Softmax')

T = np.float32
atol = 2e-3


def test_softmax1d():
    N = 20
    data = mx.random.uniform(0, 1, shape=(N, ))
    out = mobula.op.Softmax(data)
    gt = mx.nd.softmax(data)
    exp_data = mx.nd.exp(data - data.max())
    math_gt = exp_data / exp_data.sum()

    assert_almost_equal(math_gt, gt, atol=atol)
    assert_almost_equal(out, gt, atol=atol)
    # gradcheck(mobula.op.Softmax, data)


def test_softmax2d():
    def softmax2d(N, C):
        data = mx.random.uniform(0, 1, shape=(N, C))
        out = mobula.op.Softmax(data)
        gt = mx.nd.softmax(data, axis=-1)
        assert_almost_equal(out, gt, atol=atol)
        # gradcheck(mobula.op.Softmax, data, eps=1e-4)
    softmax2d(3, 10)
    softmax2d(10, 3)


def test_softmax2d_grad():
    def softmax2d_grad(N, C):
        data = mx.random.uniform(0, 1, shape=(N, C))
        data2 = data.copy()
        data.attach_grad()
        data2.attach_grad()

        dy = mx.random.uniform(0, 1, shape=(N, C)) * 1000
        with mx.autograd.record():
            out = mobula.op.Softmax(data)
        out.backward(dy)
        with mx.autograd.record():
            gt = mx.nd.softmax(data2, axis=-1)
        gt.backward(dy)
        assert_almost_equal(out, gt, atol=atol)
        assert_almost_equal(data.grad, data2.grad, atol=atol)
    softmax2d_grad(3, 10)
    softmax2d_grad(10, 3)


if __name__ == '__main__':
    test_softmax1d()
    test_softmax2d()
    test_softmax2d_grad()
