import mobula
mobula.op.load('MulElemWise')

import mxnet as mx
a = mx.nd.array([1, 2, 3])
b = mx.nd.array([4, 5, 6])
out = mx.nd.empty(a.shape)
mobula.func.mul_elemwise(a.size, a, b, out)
print(out)  # [4, 10, 18]
