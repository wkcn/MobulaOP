import sys
sys.path.append('../') # Add MobulaOP path
import mobula_op

@mobula_op.operator.register
class MyFirstOP:
    def forward(self, x, y):
        return x + y
    def backward(self, dy): 
        return [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

import mxnet as mx
a = mx.nd.array([1,2,3]) 
b = mx.nd.array([4,5,6])
c = MyFirstOP(a, b)
print (c) # [5,7,9]
