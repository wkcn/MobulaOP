import mobula_op
import mxnet as mx
import numpy as np
from mobula_op.test_utils import assert_almost_equal

def check_fc(data, weight, bias, num_hidden, no_bias, flatten):
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
            out = mobula_op.op.FullyConnected(data = data, weight = weight, num_hidden = num_hidden, no_bias = no_bias, flatten = flatten)
            out_mx = mx.nd.FullyConnected(data = data_mx, weight = weight_mx, num_hidden = num_hidden, no_bias = no_bias, flatten = flatten)
        else:
            out = mobula_op.op.FullyConnected(data = data, weight = weight, bias = bias, num_hidden = num_hidden, no_bias = no_bias, flatten = flatten)
            out_mx = mx.nd.FullyConnected(data = data_mx, weight = weight_mx, bias = bias_mx, num_hidden = num_hidden, no_bias = no_bias, flatten = flatten)
    out.backward()
    out_mx.backward()

    assert_almost_equal(out.asnumpy(), out_mx.asnumpy())
    for o, m in zip(our_data, mx_data):
        assert_almost_equal(o.grad.asnumpy(), m.grad.asnumpy())

def test_forward():
    batch_size = 5
    num_hidden = 6
    K = 10

    data = mx.nd.random.uniform(-1, 1, shape = (batch_size, K))
    data2 = mx.nd.random.uniform(-1, 1, shape = (batch_size, 4, 2, 3, K))
    bias = mx.nd.random.uniform(-1, 1, shape = (num_hidden, )) 

    for no_bias in [False, True]:
        for flatten in [False, True]:
            input_dim = K
            weight = mx.nd.random.uniform(-1, 1, shape = (num_hidden, input_dim))
            check_fc(data, weight, bias, num_hidden, no_bias, flatten)

            input_dim = data2.size // data2.shape[0] if flatten else data2.shape[-1] 
            weight = mx.nd.random.uniform(-1, 1, shape = (num_hidden, input_dim))
            check_fc(data2, weight, bias, num_hidden, no_bias, flatten)
