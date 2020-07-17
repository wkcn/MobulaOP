import time
import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal, gradcheck
import unittest

mobula.op.load('Sum')

T = np.float32


def test_sum():
    N = 1024 * 1024 * 10
    x = np.random.uniform(size=(N, ))
    y = mobula.op.Sum(x)
    assert_almost_equal(y, x.sum(), atol=1e-7)
