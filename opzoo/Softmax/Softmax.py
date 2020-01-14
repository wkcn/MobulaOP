import mobula
from mobula.const import req


# this softmax only support 2-dim input and the reduce on axis 0
@mobula.op.register
class Softmax1D:
    def forward(self, x):
        tmp = self.F.empty_like(x)
        mobula.func.softmax1d_forward(x.size, x, tmp, self.y)

    def backward(self, dy):
        raise NotImplementedError()

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 1
        return in_shape, in_shape
