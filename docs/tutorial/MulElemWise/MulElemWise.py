import mobula


@mobula.op.register
class MulElemWise:
    def forward(self, a, b):
        mobula.func.mul_elemwise(a.size, a, b, self.y)

    def backward(self, dy):
        self.dX[0][:] = self.F.multiply(dy, self.X[1])
        mobula.func.mul_elemwise(dy.size, dy, self.X[0], self.dX[1])

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]
