import sys
import os
import mxnet as mx
import numpy as np
from nose.core import TestProgram
import unittest
import mobula


def test_gpu():
    ctx = mx.gpu(0)
    mx.test_utils.set_default_context(ctx)
    assert mx.current_context() == ctx
    path = os.path.join(os.path.dirname(__file__), '../')
    TestProgram(defaultTest=path, argv=[path, '-s'], exit=False)
    assert mx.current_context() == ctx


@unittest.skipIf(len(mobula.utils.list_gpus()) < 2, 'The number of GPUs is not enough 2')
def test_multiple_gpus():
    mobula.op.load('ROIAlign')
    dtype = np.float32
    N, C, H, W = 2, 3, 4, 4

    data_cpu = mx.nd.array(
        np.arange(N * C * H * W).astype(dtype).reshape((N, C, H, W)))
    rois_cpu = mx.nd.array(np.array([[0, 1, 1, 3, 3]], dtype=dtype))

    outs = []
    for ctx in [mx.gpu(0), mx.gpu(1)]:
        data = data_cpu.as_in_context(ctx)
        rois = rois_cpu.as_in_context(ctx)
        data.attach_grad()
        with mx.autograd.record():
            # mx.nd.NDArray and mx.sym.Symbol are both available as the inputs.
            output = mobula.op.ROIAlign(data=data, rois=rois, pooled_size=(
                2, 2), spatial_scale=1.0, sampling_ratio=1)

        output.backward()
        assert output.context == ctx, (output.context, ctx)
        outs.append((output.asnumpy(), data.grad.asnumpy()))
    for a, b in zip(*outs):
        mobula.testing.assert_almost_equal(a, b)
