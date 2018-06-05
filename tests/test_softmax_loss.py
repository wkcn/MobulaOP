import mxnet as mx
import numpy as np
import mobula_op
from mobula_op.test_utils import assert_almost_equal, FLT_MIN

T = np.float32

def test_softmax_loss_forward():
    n = 5
    num_classes = 7
    data = mx.nd.random.uniform(-100, 100, shape = (n, num_classes))
    label = mx.nd.array(np.random.randint(0, num_classes, size = (n)))

    y = mobula_op.operator.SoftmaxLoss(data = data, label = label, axis = -1)
    ry = mx.nd.SoftmaxOutput(data = data, label = label, preserve_shape = True)
    assert_almost_equal(y.asnumpy(), ry.asnumpy())

    data.attach_grad()
    with mx.autograd.record():
        y, loss = mobula_op.operator.SoftmaxLoss(data = data, label = label, axis = -1, compute_loss = True)
    assert_almost_equal(y.asnumpy(), ry.asnumpy())
    y.backward()
    mx.nd.waitall()

    rlosses = np.zeros(n)
    dx = y.asnumpy()
    for i in range(n):
        a = int(label[i].asscalar())
        if a >= 0:
            rlosses[i] = -np.log(y[i, a].asscalar() + FLT_MIN)
            dx[i, a] -= 1
        else:
            dx[i, :] = 0
    rloss = rlosses.sum() / np.maximum((label.asnumpy() >= 0).sum(), 1.0)
    assert_almost_equal(loss.asnumpy(), rloss)
    assert_almost_equal(data.grad.asnumpy(), dx)

if __name__ == '__main__':
    test_softmax_loss_forward()
