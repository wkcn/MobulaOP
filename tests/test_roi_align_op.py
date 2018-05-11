import mxnet as mx
import numpy as np
import mobula

def test_roi_align_sym():
    ctx = mx.cpu(0)
    dtype = np.float32

    N, C, H, W = 2, 3, 4, 4

    data = np.arange(N*C*H*W).astype(dtype).reshape((N,C,H,W))
    rois = np.array([[0, 1, 1, 3, 3], [1, 2, 2, 3, 3]], dtype = dtype)

    data_sym = mx.sym.Variable('data')
    rois_sym = mx.sym.Variable('rois')

    output_sym = mobula.operators.ROIAlign(data = data_sym, rois = rois_sym, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1)
    output_sym = mx.sym.MakeLoss(output_sym)

    exe = output_sym.simple_bind(ctx, data = data.shape, rois = rois.shape) 
    exe.forward(data = data, rois = rois)

    res = exe.outputs[0].asnumpy()

    exe.backward()
    mx.nd.waitall()

def test_roi_align_nd():
    ctx = mx.cpu(0)
    dtype = np.float32

    N, C, H, W = 2, 3, 4, 4

    data = mx.nd.array(np.arange(N*C*H*W).astype(dtype).reshape((N,C,H,W)))
    rois = mx.nd.array(np.array([[0, 1, 1, 3, 3]], dtype = dtype))

    data.attach_grad()
    with mx.autograd.record():
        output = mobula.operators.ROIAlign(data = data, rois = rois, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1)
    output.backward()
    mx.nd.waitall()

