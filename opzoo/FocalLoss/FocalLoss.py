import mobula
from mobula.const import req
import numpy as np


@mobula.op.register
class FocalLoss:
    def __init__(self, alpha=0.25, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        if self.req[0] == req.null:
            return
        out = self.y
        out_size = np.prod(out.size()) if callable(out.size) else out.size
        alpha = self.alpha
        gamma = self.gamma
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.focal_loss_forward(out_size=out_size, alpha=alpha, gamma=gamma, logits=logits, targets=targets,
                                           outputs=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.focal_loss_forward(out_size=out_size, alpha=alpha, gamma=gamma, logits=logits, targets=targets,
                                           outputs=self.y)

    def backward(self, dy):
        assert self.req[1] == "null"
        alpha = self.alpha
        gamma = self.gamma
        logits = self.X[0]
        targets = self.X[1]
        out_size = np.prod(targets.size()) if callable(
            targets.size) else targets.size
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(self.dX[0])
            mobula.func.focal_loss_forward(out_size=out_size, alpha=alpha, gamma=gamma, logits=logits, targets=targets,
                                           outputs=out_temp)
            self.dX[0] += out_temp
        else:
            self.dX[0][:] = 0
            mobula.func.focal_loss_backward(out_size=out_size, alpha=alpha, gamma=gamma, logits=logits, targets=targets,
                                            outputs=self.dX[0])
        self.dX[0][:] = self.dX[0][:] * dy

    def infer_shape(self, in_shape):
        assert len(in_shape) == 2
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]
