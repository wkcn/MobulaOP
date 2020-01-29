import mobula
from mobula.const import req


# this softmax support 1 or 2-dim input and the reduce on axis 0
@mobula.op.register
class Softmax:
    def forward(self, x):
        if x.ndim == 2:
            N, C = x.shape
        else:
            N, C = 1, x.size
        if C > N:
            tmp = self.F.empty_like(x)
            mobula.func.softmax_channel_forward(C, N, x, tmp, self.y)
        else:
            mobula.func.softmax_batch_forward(N, C, x, self.y)

    def backward(self, dy):
        if self.req[0] == req.null:
            return
        elif self.req[0] == req.write:
            self.dx[:] = 0
        if dy.ndim == 2:
            N, C = dy.shape
        else:
            N, C = 1, dy.size
        mobula.func.softmax_backward(N, C, self.y, dy, self.dx)

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) in [1, 2]
        return in_shape, in_shape
