import mxnet as mx
import numpy as np
import mobula_op
from mobula_op.test_utils import assert_almost_equal

def test_tensordot():
    a = np.arange(60.).reshape(3,4,5).astype(np.float32)
    b = np.arange(24.).reshape(4,3,2).astype(np.float32)
    axes = ([1, 0], [0, 1])
    c = np.tensordot(a, b, axes = axes)
    d = mobula_op.math.tensordot(a, b, axes = axes) 
    assert_almost_equal(c, d)
