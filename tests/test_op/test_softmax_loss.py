import mxnet as mx
import numpy as np
import mobula_op
from mobula_op.test_utils import assert_almost_equal, FLT_MIN

T = np.float32

def test_softmax_loss():
    n = 5
    num_classes = 7
    data_np = np.random.random((n, num_classes)).astype(T)
    label_np = np.random.randint(-1, num_classes, size = (n)).astype(T)

    data_mx = mx.nd.array(data_np)
    label_mx = mx.nd.array(label_np)

    y = mobula_op.operator.SoftmaxLoss(data = data_mx, label = label_mx, axis = -1)
    ry = mx.nd.SoftmaxOutput(data = data_mx, label = label_mx, preserve_shape = True)
    assert_almost_equal(y.asnumpy(), ry.asnumpy())

    data_mx.attach_grad()
    with mx.autograd.record():
        y, loss = mobula_op.operator.SoftmaxLoss(data = data_mx, label = label_mx, axis = -1, compute_loss = True)
    assert_almost_equal(y.asnumpy(), ry.asnumpy())
    y.backward()
    mx.nd.waitall()

    rlosses = np.zeros(n)
    dx = y.asnumpy()
    for i in range(n):
        a = int(label_mx[i].asscalar())
        if a >= 0:
            rlosses[i] = -np.log(y[i, a].asscalar() + FLT_MIN)
            dx[i, a] -= 1
        else:
            dx[i, :] = 0
    num_valid = np.maximum((label_np >= 0).sum(), 1.0)
    rloss = rlosses.sum() / num_valid
    dx /= num_valid
    assert_almost_equal(loss.asnumpy(), rloss)
    assert_almost_equal(data_mx.grad.asnumpy(), dx)

    # numpy
    n = 5
    num_classes = 7
    op = mobula_op.operator.SoftmaxLoss('np')
    y_np = op(data = data_np, label = label_np, axis = -1)
    assert_almost_equal(y_np, ry.asnumpy())
    y_np, loss_np = op(data = data_np, label = label_np, axis = -1, compute_loss = True)
    assert_almost_equal(y_np, ry.asnumpy())
    assert_almost_equal(loss_np, rloss)
    dx_np, label_np = op.backward()
    assert_almost_equal(dx_np, dx)

if __name__ == '__main__':
    test_softmax_loss()
