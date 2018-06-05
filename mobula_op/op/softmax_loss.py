from .register import register
import mobula_op

@register
class SoftmaxLoss:
    def __init__(self, axis = -1):
        self.axis = axis
    def forward(self, data, label):
        outer_size, middle_size, inner_size = mobula_op.func.get_3loop_size(data.shape, self.axis) 
        mobula_op.func.softmax_loss_forward(data = data, num_classes = middle_size, outer_size = outer_size, inner_size = inner_size, probs = self.y)
    def backward(self, dy):
        pass
    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]]
