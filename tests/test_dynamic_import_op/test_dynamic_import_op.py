import mobula_op
# Import Custom Operator Dynamically
import os
op_path = os.path.join(os.path.dirname(__file__), './AdditionOP')
AdditionOP = mobula_op.import_op(op_path)

import mxnet as mx

def test_dynamic_import_op():
    a = mx.nd.array([1,2,3])
    b = mx.nd.array([4,5,6])

    a.attach_grad()
    b.attach_grad()

    with mx.autograd.record():
        c = AdditionOP(a, b)

    dc = mx.nd.array([7,8,9])
    c.backward(dc)

    assert ((a + b).asnumpy() == c.asnumpy()).all()
    assert (a.grad.asnumpy() == dc.asnumpy()).all()
    assert (b.grad.asnumpy() == dc.asnumpy()).all()
