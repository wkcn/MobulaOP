import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal, gradcheck

mobula.op.load('Transpose')

T = np.float32


def test_transpose2d():
    R, C = 3, 5
    x = mx.nd.array(np.random.uniform(size=(R, C)))
    for c in [False, True]:
        op = mobula.op.Transpose2D[mx.nd.NDArray]
        y = op(x, continuous_input=c)
        assert_almost_equal(y, x.T)
