import mxnet as mx
from mxnet.gluon import nn
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('Convolution')


def test_convolution():
    N, C, H, W = 1, 2, 3, 4
    channels = 3
    kernel_size = (2, 3)
    strides = (1, 2)
    padding = (0, 1)

    x = mx.random.uniform(0, 1, shape=(N, C, H, W))
    block = nn.Conv2D(channels=channels, kernel_size=kernel_size,
                      strides=strides, padding=padding)
    block.initialize()

    block(x)

    weight = block.weight.data()
    bias = block.bias.data()

    mx_y = block(x)

    our_y = mobula.op.Conv2D(x=x, weight=weight, bias=bias, channels=channels,
                             kernel_size=kernel_size, strides=strides, padding=padding)
    assert_almost_equal(mx_y, our_y)
