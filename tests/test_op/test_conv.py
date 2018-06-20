import mobula_op
import mxnet as mx
import numpy as np
from mobula_op.test_utils import assert_almost_equal

def check_conv(data, weight, bias, kernel, stride, dilate, pad, num_filter, no_bias):
    data_mx = data.copy()
    weight_mx = weight.copy()
    bias_mx = bias.copy()

    our_data = [data, weight, bias]
    mx_data = [data_mx, weight_mx, bias_mx]

    for d in our_data + mx_data:
        d.attach_grad()

    if no_bias:
        bias = bias_mx = None

    with mx.autograd.record():
        if no_bias:
            out = mobula_op.op.Convolution(data = data, weight = weight, num_filter = num_filter,
                    no_bias = no_bias, kernel = kernel, stride = stride, dilate = dilate, pad = pad)
            out_mx = mx.nd.Convolution(data = data_mx, weight = weight_mx, num_filter = num_filter,
                    no_bias = no_bias, kernel = kernel, stride = stride, dilate = dilate, pad = pad)
        else:
            out = mobula_op.op.Convolution(data = data, weight = weight, bias = bias, num_filter = num_filter,
                    no_bias = no_bias, kernel = kernel, stride = stride, dilate = dilate, pad = pad)
            out_mx = mx.nd.Convolution(data = data_mx, weight = weight_mx, bias = bias_mx, num_filter = num_filter,
                    no_bias = no_bias, kernel = kernel, stride = stride, dilate = dilate, pad = pad)
    out.backward()
    out_mx.backward()

    assert_almost_equal(out.asnumpy(), out_mx.asnumpy())
    for o, m in zip(our_data, mx_data):
        assert_almost_equal(o.grad.asnumpy(), m.grad.asnumpy(), atol = 5e-4)

def test_forward():

    N, C, H, W = 2, 3, 10, 15
    K = 3
    num_filter = 2
    data = mx.nd.random.uniform(-1, 1, shape = (N, C, H, W))
    weight = mx.nd.random.uniform(-1, 1, shape = (num_filter, C, K, K)) 
    bias = mx.nd.random.uniform(-1, 1, shape = (num_filter, ))
    for no_bias in [False, True]:
        for stride in [(1, 1), (2, 2)]:
            for dilate in [(1, 1), (2, 2)]:
                for pad in [(0, 0), (1, 1)]:
                    check_conv(data, weight, bias, (K, K), stride, dilate, pad, num_filter, no_bias)
