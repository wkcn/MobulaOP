import sys
sys.path.append('../') # Add MobulaOP path
import mobula_op
import numpy as np

@mobula_op.operator.register(need_top_grad = False)
class ConstantOP:
    def __init__(self, constant):
        self.constant = mx.nd.array(constant)
    def forward(self):
        return self.constant
    def backward(self, dy): 
        return [0]
    def infer_shape(self, in_shape):
        return in_shape, [self.constant.shape] 
    def infer_type(self, dtypes):
        return [], [np.float32]

if __name__ == '__main__':
    import mxnet as mx
    import numpy as np
    # NDArray
    a = mx.nd.array([1,2,3]) 
    b = mx.nd.array([4,5,6])
    c = a + ConstantOP[mx.nd.NDArray](b)
    print (c) # [5,7,9]

    # Symbol
    a_sym = mx.sym.Variable('a')
    output_sym = a_sym + ConstantOP[mx.sym.Symbol](b)
    exe = output_sym.simple_bind(ctx = mx.context.current_context(), a = a.shape)
    exe.forward(a = np.array([1,2,3]))

    print (exe.outputs[0].asnumpy()) # [5,7,9]
