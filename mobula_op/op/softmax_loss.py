from .register import register
import mobula_op

@register
class SoftmaxLoss:
    def __init__(self, axis = -1, compute_loss = False):
        self.axis = axis
        self.compute_loss = compute_loss
        self.num_outputs = 2 if compute_loss else 1
        self.num_valid = 1.0
    def forward(self, data, label):
        outer_size, middle_size, inner_size = mobula_op.func.get_3loop_size(data.shape, self.axis) 
        mobula_op.func.softmax_forward(data = data, num_classes = middle_size, outer_size = outer_size, inner_size = inner_size, probs = self.y)
        self.num_valid = self.F.maximum(1.0, (label >= 0).sum())
        if self.compute_loss:
            losses = self.F.empty((outer_size, inner_size), dtype = data.dtype)
            mobula_op.func.softmax_loss_forward(probs = self.y, labels = label, num_classes = middle_size, outer_size = outer_size, inner_size = inner_size, losses = losses)
            self.Y[1][:] = losses.sum() / self.num_valid
    def backward(self, dy, *args):
        assert self.req[0] in ['write', 'inplace']
        self.dX[0][:] = 0
        outer_size, middle_size, inner_size = mobula_op.func.get_3loop_size(self.x.shape, self.axis)
        mobula_op.func.softmax_loss_backward(probs = self.y, labels = self.X[1], num_classes = middle_size, outer_size = outer_size, inner_size = inner_size, dX = self.dx)
        self.dX[0][:] /= self.num_valid
    def infer_shape(self, in_shape):
        out_shape = [in_shape[0]]
        if self.compute_loss:
            out_shape.append((1, ))
        return in_shape, out_shape
