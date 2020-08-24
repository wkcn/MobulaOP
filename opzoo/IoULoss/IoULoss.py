import mobula
from mobula.const import req
import numpy as np


@mobula.op.register
class IoULoss:
    def forward(self, preds, targets):
        if self.req[0] == req.null:
            return
        out = self.y
        out_size = preds.shape[0]
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.iou_loss_forward(
                out_size=out_size, preds=preds, targets=targets, outputs=out_temp)

            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.iou_loss_forward(
                out_size=out_size, preds=preds, targets=targets, outputs=self.y)

    def backward(self, dy):
        assert self.req[1] == req.null
        preds = self.X[0]
        targets = self.X[1]
        out_size = preds.shape[0]
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(self.dX[0])
            mobula.func.iou_loss_backward(
                out_size=out_size, preds=preds, targets=targets, outputs=out_temp)
            self.dX[0] += out_temp
        else:
            self.dX[0][:] = 0
            mobula.func.iou_loss_backward(
                out_size=out_size, preds=preds, targets=targets, outputs=self.dX[0][:])

        self.dX[0][:] = self.dX[0][:] * dy

    def infer_shape(self, in_shape):
        assert len(in_shape) == 2
        assert len(in_shape[0]) == 2
        assert in_shape[0] == in_shape[1]
        assert in_shape[0][1] == 4
        return in_shape, [(in_shape[0][0], 1), ]
