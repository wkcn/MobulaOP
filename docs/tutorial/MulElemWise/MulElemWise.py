import mobula
import numpy as np


@mobula.op.register
class MulElemWise:
    def forward(self, a, b):
        '''
            np.array([1,2,3]).size = 3
            mx.nd.array([1,2,3]).size = 3
            torch.tensor([1,2,3]).numel() = 3
        '''
        size = a.numel() if hasattr(a, 'numel') else a.size
        mobula.func.mul_elemwise(size, a, b, self.y)

    def backward(self, dy):
        if hasattr(self.F, 'multiply'):
            # np.multiply, mx.nd.multiply
            self.dX[0][:] = self.F.multiply(dy, self.X[1])
        else:
            # torch.mul
            self.dX[0][:] = self.F.mul(dy, self.X[1])
        size = dy.numel() if hasattr(dy, 'numel') else dy.size
        mobula.func.mul_elemwise(size, dy, self.X[0], self.dX[1])

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]
