import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class ROIAlign:
    def __init__(self, pooled_size, spatial_scale, sampling_ratio):
        self.pooled_size = pooled_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, data, rois):
        if self.req[0] == req.null:
            return

        out = self.y
        out_size = np.prod(out.size()) if callable(out.size) else out.size

        if self.req[0] == req.add:
            out_temp = self.F.empty_like(out)
            mobula.func.roi_align_forward(out_size, data, self.spatial_scale, data.shape[1], data.shape[
                                          2], data.shape[3], self.pooled_size[0], self.pooled_size[1], self.sampling_ratio, rois, out_temp)
            self.y[:] += out_temp
        else:
            mobula.func.roi_align_forward(out_size, data, self.spatial_scale, data.shape[1], data.shape[
                                          2], data.shape[3], self.pooled_size[0], self.pooled_size[1], self.sampling_ratio, rois, self.y)

    def backward(self, dy):
        if self.req[0] == req.null:
            return
        if self.req[0] != req.add:
            self.dX[0][:] = 0
        data, rois = self.X

        dy_size = np.prod(dy.size()) if callable(dy.size) else dy.size
        mobula.func.roi_align_backward(dy_size, dy, self.spatial_scale, data.shape[1], data.shape[2], data.shape[
                                       3], self.pooled_size[0], self.pooled_size[1], self.sampling_ratio, self.dX[0], rois)

        if self.req[1] not in [req.null, req.add]:
            self.dX[1][:] = 0

    def infer_shape(self, in_shape):
        dshape, rshape = in_shape
        assert len(dshape) == 4
        assert len(rshape) == 2
        assert rshape[1] == 5
        oshape = [rshape[0], dshape[1],
                  self.pooled_size[0], self.pooled_size[1]]
        return [dshape, rshape], [oshape]
