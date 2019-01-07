import mobula
from mobula.testing import assert_almost_equal
# Import Custom Operator Dynamically
import os
mobula.op.load('./TemplateOP', path=os.path.dirname(__file__))

import mxnet as mx
import numpy as np


def test_template_1type():
    shape = (2, 3, 4)
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        a = mx.nd.random.uniform(0, 100, shape=shape).astype(dtype)
        b = mx.nd.random.uniform(0, 100, shape=shape).astype(dtype)
        c = mx.nd.empty(shape, dtype=dtype)
        mobula.func.maximum(a.size, a, b, c)
        assert_almost_equal(mx.nd.maximum(a, b).asnumpy(), c.asnumpy())


def test_template_3type():
    shape = (2, 3, 4)
    t1, t2, t3 = np.int32, np.float32, np.float64
    a = mx.nd.random.uniform(0, 100, shape=shape).astype(t1)
    b = mx.nd.random.uniform(0, 100, shape=shape).astype(t2)
    c = mx.nd.empty(shape, dtype=t3)
    mobula.func.maximum_3type(a.size, a, b, c)
    assert_almost_equal(mx.nd.maximum(
        a.astype(t2), b).asnumpy().astype(t3), c.asnumpy())
