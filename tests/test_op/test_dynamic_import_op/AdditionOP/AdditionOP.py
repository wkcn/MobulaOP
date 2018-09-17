import mobula


@mobula.op.register
class AdditionOP:
    def forward(self, a, b):
        c = self.y
        mobula.func.addition_op_forward(a.size, a, b, c)

    def backward(self, dc):
        return [dc, dc]

    def infer_shape(self, in_shape):
        assert list(in_shape[0]) == list(in_shape[1])
        return in_shape, [in_shape[0]]
