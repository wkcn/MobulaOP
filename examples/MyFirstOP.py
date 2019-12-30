import sys
sys.path.append('../')  # Add MobulaOP path
import mobula


@mobula.op.register
class MyFirstOP:
    def forward(self, x, y):
        return x + y

    def backward(self, dy):
        return [dy, dy]

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


try:
    import mxnet as mx
    print('MXNet:')
    print('mx.nd.NDArray:')
    a = mx.nd.array([1, 2, 3])
    b = mx.nd.array([4, 5, 6])
    c = MyFirstOP(a, b)
    print('a + b = c\n{} + {} = {}\n'.format(a.asnumpy(),
                                             b.asnumpy(), c.asnumpy()))  # [5, 7, 9]

    if hasattr(mx, 'numpy'):
        # MXNet NumPy-compatible API
        print('mx.np.ndarray:')
        a = mx.np.array([1, 2, 3])
        b = mx.np.array([4, 5, 6])
        c = MyFirstOP(a, b)
        print('a + b = c\n{} + {} = {}\n'.format(a.asnumpy(),
                                                 b.asnumpy(), c.asnumpy()))  # [5, 7, 9]
except ImportError:
    pass

try:
    import numpy as np
    print('NumPy:')
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    op = MyFirstOP[np.ndarray]()
    c = op(a, b)
    print('a + b = c\n{} + {} = {}\n'.format(a, b, c))  # [5, 7, 9]
except ImportError:
    pass

try:
    import torch
    print('PyTorch:')
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = MyFirstOP(a, b)
    print('a + b = c\n{} + {} = {}\n'.format(a, b, c))  # [5, 7, 9]
except ImportError:
    pass

try:
    import cupy as cp
    print('CuPy:')
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    op = MyFirstOP[cp.ndarray]()
    c = op(a, b)
    print('a + b = c\n{} + {} = {}\n'.format(a, b, c))  # [5, 7, 9]
except ImportError:
    pass
