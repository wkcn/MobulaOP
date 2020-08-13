import mobula

mobula.op.load('FocalLoss')
import mxnet as mx
import mxnet.autograd as ag
from mobula.testing import assert_almost_equal


def BCEFocalLoss(x, target, alpha=.25, gamma=2):
    p = x.sigmoid()
    loss = alpha * target * ((1 - p) ** gamma) * mx.nd.log(p)
    loss = loss + (1 - alpha) * (1 - target) * (p ** gamma) * mx.nd.log(1 - p)
    return -loss


N = 42


def test_FocalLoss_mx_cpu():
    ctx = mx.cpu()
    x = mx.nd.random.randn(N, N, dtype="float64", ctx=ctx)
    y = mx.nd.random.randn(N, N, dtype="float64", ctx=ctx)
    x1 = x.copy()
    y1 = y.copy()

    x.attach_grad()
    x1.attach_grad()

    with ag.record():
        fl = BCEFocalLoss(x, y, alpha=.25, gamma=2)
        fl_mobula = mobula.op.FocalLoss(
            alpha=.25, gamma=2, logits=x1, targets=y1)
        fl.backward()
        fl_mobula.backward()

    assert_almost_equal(x.grad.asnumpy(), x1.grad.asnumpy())
    assert_almost_equal(fl.asnumpy(), fl_mobula.asnumpy())


def test_FocalLoss_mx_cuda():
    if len(mobula.utils.list_gpus()) == 0:
        return
    ctx = mx.gpu()
    x = mx.nd.random.randn(N, N, dtype="float64", ctx=ctx)
    y = mx.nd.random.randn(N, N, dtype="float64", ctx=ctx)
    x1 = x.copy()
    y1 = y.copy()

    x.attach_grad()
    x1.attach_grad()

    with ag.record():
        fl = BCEFocalLoss(x, y, alpha=.25, gamma=2)
        fl_mobula = mobula.op.FocalLoss(
            alpha=.25, gamma=2, logits=x1, targets=y1)
        fl.backward()
        fl_mobula.backward()

    assert_almost_equal(x.grad, x1.grad)
    assert_almost_equal(fl, fl_mobula)


if __name__ == '__main__':
    test_FocalLoss_mx_cpu()
    test_FocalLoss_mx_cuda()
