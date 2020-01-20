import time
import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('Softmax')

T = np.float32
atol = 1e-3


def test_softmax1d():
    N = 20
    data = mx.random.uniform(0, 1, shape=(N, ))
    out = mobula.op.Softmax(data)
    gt = mx.nd.softmax(data)
    exp_data = mx.nd.exp(data - data.max())
    math_gt = exp_data / exp_data.sum()

    print("===Softmax1D===")
    print(out, gt)
    print(out.sum(), gt.sum())

    assert_almost_equal(math_gt, gt, atol=atol)
    assert_almost_equal(out, gt, atol=atol)


def test_softmax2d():
    def softmax2d(N, C):
        print("===Softmax2D===", N, C)
        data = mx.random.uniform(0, 1, shape=(N, C))
        out = mobula.op.Softmax(data)
        gt = mx.nd.softmax(data, axis=-1)
        print(out, gt)
        assert_almost_equal(out, gt, atol=atol)
    softmax2d(3, 10)
    softmax2d(10, 3)


if __name__ == '__main__':
    test_softmax1d()
    test_softmax2d()
