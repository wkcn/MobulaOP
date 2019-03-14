import mobula
from mobula.testing import assert_almost_equal
# Import Custom Operator Dynamically
import os
mobula.op.load('./AdditionOP', path=os.path.dirname(__file__))
AdditionOP = mobula.op.AdditionOP

import mxnet as mx


def test_addition():
    a = mx.nd.array([1, 2, 3])
    b = mx.nd.array([4, 5, 6])

    a.attach_grad()
    b.attach_grad()

    with mx.autograd.record():
        c = AdditionOP(a, b)

    dc = mx.nd.array([7, 8, 9])
    c.backward(dc)

    assert_almost_equal(a + b, c)
    assert_almost_equal(a.grad, dc)
    assert_almost_equal(b.grad, dc)
