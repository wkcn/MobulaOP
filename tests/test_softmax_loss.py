import mxnet as mx
import numpy as np
import mobula_op

T = np.float32

def test_softmax_loss_forward():
    n = 5
    num_classes = 7
    data = mx.nd.random.uniform(-100, 100, shape = (n, num_classes))
    label = mx.nd.array(np.random.randint(0, num_classes, size = (n)))

    y = mobula_op.operator.SoftmaxLoss(data = data, label = label, axis = -1)
    ry = mx.nd.SoftmaxOutput(data = data, label = label, preserve_shape = True)
    assert (y.asnumpy() == ry.asnumpy()).all(), (y, ry)

if __name__ == '__main__':
    test_softmax_loss_forward()
