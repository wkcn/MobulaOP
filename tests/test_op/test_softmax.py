import time
import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('Softmax')


def test_softmax():
    T = np.float32

    N = 20
    data = mx.random.uniform(0, 1, shape=(N, ))
    out = mobula.op.Softmax1D(data)
    gt = mx.nd.softmax(data)
    exp_data = mx.nd.exp(data - data.max())
    math_gt = exp_data / exp_data.sum()

    atol = 1e-3
    assert_almost_equal(math_gt, gt, atol=atol)
    assert_almost_equal(out, gt, atol=atol)
