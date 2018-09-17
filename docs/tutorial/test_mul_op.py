import mobula
mobula.op.load('MulElemWise')

import mxnet as mx
a = mx.nd.array([1, 2, 3])
b = mx.nd.array([4, 5, 6])

print('==========')
print('MXNet NDArray')
a.attach_grad()
b.attach_grad()
with mx.autograd.record():
    c = mobula.op.MulElemWise(a, b)
    c.backward()
    print(c)  # [4, 10, 18]
    print('a.grad = {}'.format(a.grad.asnumpy()))  # [4, 5, 6]
    print('b.grad = {}'.format(b.grad.asnumpy()))  # [1, 2, 3]

print('==========')
print('MXNet Symbol')
a_sym = mx.sym.Variable('a')
b_sym = mx.sym.Variable('b')
c_sym = mobula.op.MulElemWise(a_sym, b_sym)
exe = c_sym.simple_bind(ctx=mx.context.current_context(), a=a.shape, b=b.shape)
exe.forward(a=a, b=b)
print(exe.outputs[0])

print('==========')
print('NumPy')
import numpy as np
a_np = np.array([1, 2, 3])
b_np = np.array([4, 5, 6])
op = mobula.op.MulElemWise[np.ndarray]()
c_np = op(a_np, b_np)
print(c_np)

print('==========')
print('MXNet Gluon')


class MulElemWiseBlock(mx.gluon.nn.HybridBlock):
    def hybrid_forward(self, F, a, b):
        return mobula.op.MulElemWise(a, b)


block = MulElemWiseBlock()
print(block(a, b))
