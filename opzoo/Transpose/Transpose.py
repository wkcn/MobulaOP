import mobula
from mobula.const import req


@mobula.op.register
class Transpose2D:
    def __init__(self, continuous_input=True):
        self.func = mobula.func.transpose_2d_ci if continuous_input else mobula.func.transpose_2d_co

    def forward(self, x):
        assert self.req[0] in [req.write, req.inplace]
        R, C = x.shape
        self.func(x.size, x, R, C, self.y)

    def backward(self, dy):
        assert self.req[0] in [req.write, req.inplace]
        R, C = self.x.shape
        self.func(self.x.size, dy, C, R, self.dx)

    def infer_shape(self, in_shape):
        assert len(in_shape) == 1
        assert len(in_shape[0]) == 2
        R, C = in_shape[0]
        return in_shape, [(C, R)]
