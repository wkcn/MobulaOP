import mxnet as mx
import numpy as np
import mobula

def bilinear_interpolate(bottom, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return 0.0, []
    x = max(0.0, x)
    y = max(0.0, y)
    x_low = int(x)
    y_low = int(y)
    if x_low >= width - 1:
        x_low = x_high = width - 1
        x = x_low
    else:
        x_high = x_low + 1

    if y_low >= height - 1:
        y_low = y_high = height - 1
        y = y_low
    else:
        y_high = y_low + 1

    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    v1 = bottom[y_low, x_low]
    v2 = bottom[y_low, x_high]
    v3 = bottom[y_high, x_low]
    v4 = bottom[y_high, x_high]

    '''
    ----------->x
    |hx hy | lx hy
    |------+------
    |hx ly | lx ly
    V
    y

    v1|v2
    --+--
    v3|v4
    '''
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    grad = [(y_low, x_low, w1), (y_low, x_high, w2),
            (y_high, x_low, w3), (y_high, x_high, w4)
           ]
    return val, grad

def roialign_forward_backward(data, rois, pooled_size, spatial_scale, sampling_ratio, dy):
    N, C, H, W = data.shape
    R = rois.shape[0]
    PH, PW = pooled_size
    assert len(rois.shape) == 2
    assert rois.shape[1] == 5

    out = np.zeros((R, C, PH, PW))
    dx = np.zeros_like(data)
    drois = np.zeros_like(rois)

    for r in range(R):
        batch_ind = int(rois[r, 0])
        sw, sh, ew, eh = rois[r, 1:5] * spatial_scale
        roi_w = max(ew - sw, 1.0)
        roi_h = max(eh - sh, 1.0)
        bin_h = roi_h * 1.0 / PH
        bin_w = roi_w * 1.0 / PW
        bdata = data[batch_ind]
        if sampling_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = sampling_ratio
        else:
            roi_bin_grid_h = int(np.ceil(roi_h * 1.0 / PH))
            roi_bin_grid_w = int(np.ceil(roi_w * 1.0 / PW))
        count = roi_bin_grid_h * roi_bin_grid_w
        for c in range(C):
            for ph in range(PH):
                for pw in range(PW):
                    val = 0.0
                    for iy in range(roi_bin_grid_h):
                        y = sh + ph * bin_h + (iy + 0.5) * bin_h / roi_bin_grid_h
                        for ix in range(roi_bin_grid_w):
                            x = sw + pw * bin_w + (ix + 0.5) * bin_w / roi_bin_grid_w
                            v, g = bilinear_interpolate(bdata[c], H, W, y, x)
                            val += v
                            # compute grad
                            for qy, qx, qw in g:
                                dx[batch_ind, c, qy, qx] += dy[r, c, ph, pw] * qw * 1.0 / count

                    out[r, c, ph, pw] = val * 1.0 / count
    return out, [dx, drois]

def test_roi_align_sym():
    ctx = mx.cpu(0)
    dtype = np.float32

    N, C, H, W = 2, 3, 4, 4

    data = np.arange(N*C*H*W).astype(dtype).reshape((N,C,H,W))
    rois = np.array([[0, 1, 1, 3, 3], [1, 2, 2, 3, 3]], dtype = dtype)

    data_sym = mx.sym.Variable('data')
    rois_sym = mx.sym.Variable('rois')

    output_sym = mobula.operator.ROIAlign(data = data_sym, rois = rois_sym, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1)
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
        output = mobula.operator.ROIAlign(data = data, rois = rois, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1)
    output.backward()
    mx.nd.waitall()

def test_roi_align_value():
    ctx = mx.cpu(0)
    dtype = np.float32

    dlen = 224
    N, C, H, W = 5, 3, 16, 16
    assert H == W
    R = 7
    pooled_size = (3, 4)

    spatial_scale = H * 1.0 / dlen
    sampling_ratio = 0
    data = mx.nd.array(np.arange(N*C*W*H).reshape((N,C,H,W)), dtype = dtype)
    # data = mx.nd.random.uniform(0, 1, (N, C, H, W), dtype = dtype)
    center_xy = mx.nd.random.uniform(0, dlen, (R, 2), dtype = dtype)
    wh = mx.nd.random.uniform(0, dlen, (R, 2), dtype = dtype)
    batch_ind = mx.nd.array(np.random.randint(0, N, size = (R,1)))
    pos = mx.nd.concat(center_xy - wh / 2, center_xy + wh / 2, dim = 1)
    rois = mx.nd.concat(batch_ind, pos, dim = 1)

    data.attach_grad()
    rois.attach_grad()
    with mx.autograd.record():
        output = mobula.operator.ROIAlign(data = data, rois = rois, pooled_size = pooled_size, spatial_scale = spatial_scale, sampling_ratio = sampling_ratio)
    dy = mx.nd.random.uniform(-1, 1, (R, C) + pooled_size, dtype = dtype)
    output.backward(dy)
    real_output, [dx, drois] = roialign_forward_backward(data.asnumpy(), rois.asnumpy(), pooled_size, spatial_scale, sampling_ratio, dy.asnumpy())
    assert np.allclose(output.asnumpy(), real_output)
    # It seems that the precision between Cfloat and Pyfloat is different.
    assert np.allclose(data.grad.asnumpy(), dx, atol = 1e-6), np.abs(data.grad.asnumpy() - dx).max()
    assert np.allclose(rois.grad.asnumpy(), drois)

if __name__ == '__main__':
    test_roi_align_value()
    test_roi_align_sym()
    test_roi_align_nd()
