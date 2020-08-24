import sys
import mobula
import mxnet as mx
import numpy as np
import mxnet.autograd as ag
from mobula.testing import assert_almost_equal
mobula.op.load('IoULoss')


class IoULoss(mx.gluon.nn.Block):
    def __init__(self):
        super(IoULoss, self).__init__()

    def max(self, *args):
        if len(args) == 1:
            return args[0]
        maximum = args[0]
        for arg in args[1:]:
            maximum = mx.nd.maximum(maximum, arg)
        return maximum

    def forward(self, prediction, target):
        assert prediction.shape[1] == 4
        assert target.shape[1] == 4
        target = mx.nd.log(target)

        l, t, r, b = 0, 1, 2, 3  # l, t, r, b
        tl = target[:, t] + target[:, l]
        tr = target[:, t] + target[:, r]
        bl = target[:, b] + target[:, l]
        br = target[:, b] + target[:, r]
        tl_hat = prediction[:, t] + prediction[:, l]
        tr_hat = prediction[:, t] + prediction[:, r]
        bl_hat = prediction[:, b] + prediction[:, l]
        br_hat = prediction[:, b] + prediction[:, r]

        x_t_i = mx.nd.minimum(target[:, t], prediction[:, t])
        x_b_i = mx.nd.minimum(target[:, b], prediction[:, b])
        x_l_i = mx.nd.minimum(target[:, l], prediction[:, l])
        x_r_i = mx.nd.minimum(target[:, r], prediction[:, r])

        tl_i = x_t_i + x_l_i
        tr_i = x_t_i + x_r_i
        bl_i = x_b_i + x_l_i
        br_i = x_b_i + x_r_i

        max_v = self.max(tl, tr, bl, br, tl_hat, tr_hat,
                         bl_hat, br_hat, tl_i, tr_i, bl_i, br_i)
        I = mx.nd.exp(tl_i - max_v) + mx.nd.exp(tr_i - max_v) + \
            mx.nd.exp(bl_i - max_v) + mx.nd.exp(br_i - max_v)
        X = mx.nd.exp(tl - max_v) + mx.nd.exp(tr - max_v) + \
            mx.nd.exp(bl - max_v) + mx.nd.exp(br - max_v)
        X_hat = mx.nd.exp(tl_hat - max_v) + mx.nd.exp(tr_hat - max_v) + \
            mx.nd.exp(bl_hat - max_v) + mx.nd.exp(br_hat - max_v)
        I_over_U = I / (X + X_hat - I)
        return -(I_over_U).log()


N = 42


def test_IoULoss_mx(ctx):
    x = mx.nd.random.uniform(1, 3, shape=(N, 4), dtype="float64", ctx=ctx)
    y = mx.nd.random.uniform(np.exp(4), np.exp(
        5), shape=(N, 4), dtype="float64", ctx=ctx)

    x1 = x.copy()
    y1 = y.copy()

    x.attach_grad()
    x1.attach_grad()

    with ag.record():
        loss = IoULoss()(x, y)
        loss_mobula = mobula.op.IoULoss(x1, y1).squeeze()
        loss.backward()
        loss_mobula.backward()
    mx.nd.waitall()
    assert_almost_equal(x.grad.asnumpy(), x1.grad.asnumpy())
    assert_almost_equal(loss.asnumpy(), loss_mobula.asnumpy())

    x = mx.nd.random.uniform(3, 5, shape=(N, 4), dtype="float64", ctx=ctx)
    y = mx.nd.random.uniform(np.exp(1), np.exp(
        2), shape=(N, 4), dtype="float64", ctx=ctx)

    x1 = x.copy()
    y1 = y.copy()

    x.attach_grad()
    x1.attach_grad()

    with ag.record():
        loss = IoULoss()(x, y)
        loss_mobula = mobula.op.IoULoss(x1, y1).squeeze()
        loss.backward()
        loss_mobula.backward()
    mx.nd.waitall()
    assert_almost_equal(x.grad.asnumpy(), x1.grad.asnumpy())
    assert_almost_equal(loss.asnumpy(), loss_mobula.asnumpy())

    x = mx.nd.random.uniform(1, 5, shape=(N, 4), dtype="float64", ctx=ctx)
    y = mx.nd.random.uniform(np.exp(1), np.exp(
        5), shape=(N, 4), dtype="float64", ctx=ctx)

    x1 = x.copy()
    y1 = y.copy()

    x.attach_grad()
    x1.attach_grad()

    with ag.record():
        loss = IoULoss()(x, y)
        loss_mobula = mobula.op.IoULoss(x1, y1).squeeze()
        loss.backward()
        loss_mobula.backward()
    mx.nd.waitall()
    assert_almost_equal(x.grad.asnumpy(), x1.grad.asnumpy())
    assert_almost_equal(loss.asnumpy(), loss_mobula.asnumpy())


def test_IoULoss_mx_cpu():
    ctx = mx.cpu()
    test_IoULoss_mx(ctx)


def test_IoULoss_mx_cuda():
    if len(mobula.utils.list_gpus()) == 0:
        return
    ctx = mx.gpu()
    test_IoULoss_mx(ctx)


if __name__ == '__main__':
    test_IoULoss_mx_cpu()
    test_IoULoss_mx_cuda()
