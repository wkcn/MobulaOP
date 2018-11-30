# Use ROIAlign operator
import sys
sys.path.append('../')  # Add MobulaOP path
import mxnet as mx
import numpy as np
import mobula
# Load ROIAlign Module
mobula.op.load('ROIAlign')

ctx = mx.cpu(0)
dtype = np.float32
N, C, H, W = 2, 3, 4, 4

data = mx.nd.array(
    np.arange(N * C * H * W).astype(dtype).reshape((N, C, H, W)))
rois = mx.nd.array(np.array([[0, 1, 1, 3, 3]], dtype=dtype))

data.attach_grad()
with mx.autograd.record():
    # mx.nd.NDArray and mx.sym.Symbol are both available as the inputs.
    output = mobula.op.ROIAlign(data=data, rois=rois, pooled_size=(
        2, 2), spatial_scale=1.0, sampling_ratio=1)

output.backward()

print(output.asnumpy(), data.grad.asnumpy())
