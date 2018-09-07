import mobula
from mobula.test_utils import assert_almost_equal
# Import Custom Operator Dynamically
import os
mobula.op.load('./TemplateOP', path=os.path.dirname(__file__))

import mxnet as mx
import numpy as np

def test_maximum():
    shape = (2,3,4)
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        a = mx.nd.random.uniform(0, 100, shape=shape).astype(dtype)
        b = mx.nd.random.uniform(0, 100, shape=shape).astype(dtype)
        c = mx.nd.empty(shape, dtype=dtype)
        mobula.func.maximum(a.size, a, b, c)
        assert_almost_equal(mx.nd.maximum(a, b).asnumpy(), c.asnumpy())
