import mobula
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

    assert ((a + b).asnumpy() == c.asnumpy()).all()
    assert (a.grad.asnumpy() == dc.asnumpy()).all()
    assert (b.grad.asnumpy() == dc.asnumpy()).all()
