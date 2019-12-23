import time
import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('ROIAlign')

T = np.float32


def test_roi_align_sym(op, times):
    dtype = np.float32

    N, C, H, W = 2, 1024, 14, 14
    num_rois = 512

    data = np.arange(N * C * H * W).astype(dtype).reshape((N, C, H, W))
    rois = np.empty((num_rois, 5))
    for i in range(num_rois):
        rois[i] = [i % N, 0, 0, i % H, i % W]

    data_sym = mx.sym.Variable('data')
    rois_sym = mx.sym.Variable('rois')

    output_sym = mobula.op.ROIAlign(data=data_sym, rois=rois_sym, pooled_size=(
        2, 2), spatial_scale=1.0, sampling_ratio=1)
    output_sym = mx.sym.MakeLoss(output_sym)

    exe = output_sym.simple_bind(
        ctx=mx.context.current_context(), data=data.shape, rois=rois.shape)
    exe.forward(data=data, rois=rois)
    res = exe.outputs[0].asnumpy()
    for t in range(times):
        if t == 1:
            tic = time.time()
        exe.forward(data=data, rois=rois)
        res = exe.outputs[0].asnumpy()
        exe.backward()
        mx.nd.waitall()
    cost = time.time() - tic
    return cost


def test_roi_align():
    TIMES = 50
    for op in [mobula.op.ROIAlign, mx.sym.contrib.ROIAlign]:
        cost = test_roi_align_sym(op, TIMES)
        print(op, cost)


if __name__ == '__main__':
    print("===cpu===")
    test_roi_align()

    if mobula.utils.list_gpus():
        print("===gpu===")
        ctx = mx.gpu(0)
        mx.test_utils.set_default_context(ctx)
        test_roi_align()
