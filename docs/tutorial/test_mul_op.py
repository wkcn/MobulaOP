import mobula
mobula.op.load('MulElemWise')


def test_numpy():
    print('==========')
    print('NumPy')
    a_np = np.array([1, 2, 3])
    b_np = np.array([4, 5, 6])
    op = mobula.op.MulElemWise[np.ndarray]()
    c_np = op(a_np, b_np)
    print(c_np)
    print('gradients:', op.backward())


def test_cupy():
    print('==========')
    print('CuPy')
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    op = mobula.op.MulElemWise[cp.ndarray]()
    c = op(a, b)
    print(c)  # [4, 10, 18]
    print('gradients:', op.backward())


def test_mxnet_ndarray():
    print('==========')
    print('MXNet NDArray')
    a = mx.nd.array([1, 2, 3])
    b = mx.nd.array([4, 5, 6])
    a.attach_grad()
    b.attach_grad()
    with mx.autograd.record():
        c = mobula.op.MulElemWise(a, b)
        c.backward()
    print(c)  # [4, 10, 18]
    print('gradients:', a.grad, b.grad)


def test_mxnet_symbol():
    print('==========')
    print('MXNet Symbol')
    a = mx.nd.array([1, 2, 3])
    b = mx.nd.array([4, 5, 6])
    a_sym = mx.sym.Variable('a')
    b_sym = mx.sym.Variable('b')
    c_sym = mobula.op.MulElemWise(a_sym, b_sym)
    exe = c_sym.simple_bind(
        ctx=mx.context.current_context(), a=a.shape, b=b.shape)
    exe.forward(a=a, b=b)
    print(exe.outputs[0])


def test_mxnet_gluon():
    print('==========')
    print('MXNet Gluon')
    a = mx.nd.array([1, 2, 3])
    b = mx.nd.array([4, 5, 6])

    class MulElemWiseBlock(mx.gluon.nn.HybridBlock):
        def hybrid_forward(self, F, a, b):
            return mobula.op.MulElemWise(a, b)
    block = MulElemWiseBlock()
    print(block(a, b))


def test_pytorch():
    print('==========')
    print('PyTorch')
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = torch.tensor([4., 5., 6.], requires_grad=True)
    c = mobula.op.MulElemWise(a, b)  # c = a + b
    c.sum().backward()
    print(c)
    print('gradients:', a.grad, b.grad)


if __name__ == '__main__':
    import numpy as np
    test_numpy()

    try:
        import cupy as cp
        test_cupy()
    except ImportError:
        pass

    try:
        import mxnet as mx
        test_mxnet_ndarray()
        test_mxnet_symbol()
        test_mxnet_gluon()
    except ImportError:
        pass

    try:
        import torch
        test_pytorch()
    except ImportError:
        pass
