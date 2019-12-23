import mobula


@mobula.op.register
class EmptyOP:
    def forward(self, a):
        mobula.func.empty_forward(a.size, a, self.y)

    def backward(self, dy):
        mobula.func.empty_forward(a.size, a, self.dx)

    def infer_shape(self, in_shape):
        assert len(in_shape) == 1
        return in_shape, in_shape
