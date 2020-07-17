import mxnet as mx
from mxnet.gluon import nn
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('Convolution')


def test_convolution():
    N, C, H, W = 2, 2, 3, 4
    channels = 3
    kernel_size = (2, 3)
    strides = (1, 2)
    padding = (0, 1)

    x = mx.random.uniform(0, 1, shape=(N, C, H, W))
    our_x = x.copy()
    block = nn.Conv2D(channels=channels, kernel_size=kernel_size,
                      strides=strides, padding=padding)
    block.initialize()

    y = block(x)
    out_grad = mx.random.uniform(0, 1, shape=y.shape)

    weight = block.weight.data()
    bias = block.bias.data()

    x.attach_grad()
    with mx.autograd.record():
        mx_y = block(x)
    mx_y.backward(out_grad)

    our_x.attach_grad()
    our_weight = weight.copy()
    our_weight.attach_grad()
    our_bias = bias.copy()
    our_bias.attach_grad()

    with mx.autograd.record():
        our_y = mobula.op.Conv2D(x=our_x, weight=our_weight, bias=our_bias, channels=channels,
                                 kernel_size=kernel_size, strides=strides, padding=padding)
    our_y.backward(out_grad)

    atol = 1e-6
    assert_almost_equal(mx_y, our_y, atol=atol)
    assert_almost_equal(x.grad, our_x.grad, atol=atol)
    assert_almost_equal(weight.grad, our_weight.grad, atol=atol)
    assert_almost_equal(bias.grad, our_bias.grad, atol=atol)


if __name__ == '__main__':
    test_convolution()
