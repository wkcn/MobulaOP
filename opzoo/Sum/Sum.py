import mobula


@mobula.op.register
class Sum:
    def forward(self, x):
        tmp = self.F.empty_like(x)
        mobula.func.sum(x.size, x, tmp)
        self.assign(self.y, self.req[0], tmp[0])

    def backward(self, dy):
        self.assign(self.dx, self.req[0], dy)

    def infer_shape(self, in_shape):
        assert len(in_shape) == 1
        assert len(in_shape[0]) == 1
        return in_shape, [(1, )]
