""" Example for using TVM generated function """
import sys
sys.path.append('../')  # Add MobulaOP path
import mobula

import tvm
import topi
from tvm.contrib.mxnet import to_mxnet_func
from tvm.contrib.dlpack import to_pytorch_func


def get_tvm_add():
    # define compute
    n = tvm.var('n')
    A = tvm.placeholder(n, name='A', dtype='float32')
    B = tvm.placeholder(n, name='B', dtype='float32')
    C = tvm.compute((n,), lambda i: A[i] + B[i], name='C')

    # build function (with parallel support)
    with tvm.target.create('llvm'):
        s = topi.generic.schedule_injective([C])
        func_cpu = tvm.build(s, [A, B, C])

    with tvm.target.create('cuda'):
        s = topi.generic.schedule_injective([C])
        func_gpu = tvm.build(s, [A, B, C])

    return func_cpu, func_gpu


@mobula.op.register
class TVMAddOp:
    def __init__(self):
        func_cpu, func_gpu = get_tvm_add()

        self.func = {
            'mx': {
                'cpu': to_mxnet_func(func_cpu, const_loc=[0, 1]),
                'gpu': to_mxnet_func(func_gpu, const_loc=[0, 1]),
            },
            'th': {
                'cpu': to_pytorch_func(func_cpu),
                'gpu': to_pytorch_func(func_gpu),
            }
        }

    def forward(self, x, y):
        b = mobula.glue.backend.get_var_backend(x)
        backend = (b.__name__.split('.')[-1])
        device_type = 'cpu' if b.dev_id(x) is None else 'gpu'

        self.func[backend][device_type](x, y, self.Y[0])

    def backward(self, dy):
        return [dy, dy]

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


try:
    import mxnet as mx
    print('===== MXNet =====')
    for ctx in [mx.cpu(), mx.gpu()]:
        print(ctx)
        a = mx.nd.array([1.0, 2.0, 3.0], ctx=ctx)
        b = mx.nd.array([4.0, 5.0, 6.0], ctx=ctx)
        c = TVMAddOp(a, b)
        print('a + b = c\n{} + {} = {}\n'.format(a.asnumpy(),
                                                 b.asnumpy(), c.asnumpy()))  # [5.0, 7.0, 9.0]
except ImportError:
    pass

try:
    import torch
    print('===== PyTorch =====')
    for device in [torch.device('cpu'), torch.device('cuda')]:
        print(device)
        a = torch.tensor([1.0, 2.0, 3.0], device=device)
        b = torch.tensor([4.0, 5.0, 6.0], device=device)
        c = TVMAddOp(a, b)
        print('a + b = c\n{} + {} = {}\n'.format(a, b, c))  # [5.0, 7.0, 9.0]
except ImportError:
    pass
