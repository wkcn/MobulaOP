import mobula

@mobula.op.register
class mul_elemwise:
    def forward(self, a, b):
        mobula.func.mul_elemwise(a.size, a, b, self.y)
    def backward(self, dy):
        mobula.func.mul_elemwise(dy.size, dy, b, self.dX[0]) 
        mobula.func.mul_elemwise(dy.size, dy, a, self.dX[1]) 
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]
