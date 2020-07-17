import mxnet as mx
import numpy as np
import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('ROIAlign')

T = np.float32


def bilinear_interpolate(bottom, height, width, y, x):
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return T(0.0), []
    x = T(max(0.0, x))
    y = T(max(0.0, y))
    x_low = int(x)
    y_low = int(y)
    if x_low >= width - 1:
        x_low = x_high = width - 1
        x = T(x_low)
    else:
        x_high = x_low + 1

    if y_low >= height - 1:
        y_low = y_high = height - 1
        y = T(y_low)
    else:
        y_high = y_low + 1

    ly = y - T(y_low)
    lx = x - T(x_low)
    hy = T(1.0) - ly
    hx = T(1.0) - lx

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

    assert w1.dtype == T
    assert w2.dtype == T
    assert w3.dtype == T
    assert w4.dtype == T

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    assert val.dtype == T
    grad = [(y_low, x_low, w1), (y_low, x_high, w2),
            (y_high, x_low, w3), (y_high, x_high, w4)
            ]
    return val, grad


def bilinear_interpolate_gradient(height, width, y, x):
    '''
    return:
        w1, w2, w3, w4
        x_low, x_high, y_low, y_high
    '''
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return T(0), T(0), T(0), T(0), -1, -1, -1, -1
    x = T(max(0.0, x))
    y = T(max(0.0, y))
    x_low = int(x)
    y_low = int(y)
    if x_low >= width - 1:
        x_low = x_high = width - 1
        x = T(x_low)
    else:
        x_high = x_low + 1

    if y_low >= height - 1:
        y_low = y_high = height - 1
        y = T(y_low)
    else:
        y_high = y_low + 1

    ly = y - T(y_low)
    lx = x - T(x_low)
    hy = T(1.0) - ly
    hx = T(1.0) - lx

    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx
    return w1, w2, w3, w4, x_low, x_high, y_low, y_high


def roialign_forward_backward(data, rois, pooled_size, spatial_scale, sampling_ratio, dy):
    N, C, H, W = data.shape
    R = rois.shape[0]
    PH, PW = pooled_size
    assert len(rois.shape) == 2
    assert rois.shape[1] == 5
    assert data.dtype == T
    assert rois.dtype == T

    out = np.zeros((R, C, PH, PW), dtype=T)
    dx = np.zeros_like(data)
    drois = np.zeros_like(rois)

    for r in range(R):
        batch_ind = int(rois[r, 0])
        sw, sh, ew, eh = rois[r, 1:5] * T(spatial_scale)
        roi_w = T(max(ew - sw, 1.0))
        roi_h = T(max(eh - sh, 1.0))
        bin_h = roi_h / T(PH)
        bin_w = roi_w / T(PW)
        bdata = data[batch_ind]
        if sampling_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = sampling_ratio
        else:
            roi_bin_grid_h = int(np.ceil(roi_h / T(PH)))
            roi_bin_grid_w = int(np.ceil(roi_w / T(PW)))
        count = T(roi_bin_grid_h * roi_bin_grid_w)
        for c in range(C):
            for ph in range(PH):
                for pw in range(PW):
                    val = T(0.0)
                    for iy in range(roi_bin_grid_h):
                        y = sh + T(ph) * bin_h + (T(iy) + T(0.5)) * \
                            bin_h / T(roi_bin_grid_h)
                        for ix in range(roi_bin_grid_w):
                            x = sw + T(pw) * bin_w + (T(ix) + T(0.5)) * \
                                bin_w / T(roi_bin_grid_w)
                            v, g = bilinear_interpolate(bdata[c], H, W, y, x)
                            assert v.dtype == T
                            val += v
                            # compute grad
                            for qy, qx, qw in g:
                                assert qw.dtype == T
                                dx[batch_ind, c, qy, qx] += dy[r,
                                                               c, ph, pw] * qw / count

                    out[r, c, ph, pw] = val / count
    assert out.dtype == T, out.dtype
    return out, [dx, drois]


def roialign_backward(bottom_diff, rois, pooled_size, spatial_scale, sampling_ratio, top_diff):
    N, C, H, W = bottom_diff.shape
    R = rois.shape[0]
    PH, PW = pooled_size
    assert len(rois.shape) == 2
    assert rois.shape[1] == 5
    assert rois.dtype == T

    for r in range(R):
        batch_ind = int(rois[r, 0])
        sw, sh, ew, eh = rois[r, 1:5] * T(spatial_scale)
        roi_w = max(ew - sw, T(1.0))
        roi_h = max(eh - sh, T(1.0))
        bin_h = roi_h / T(PH)
        bin_w = roi_w / T(PW)
        if sampling_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = int(sampling_ratio)
        else:
            roi_bin_grid_h = int(np.ceil(roi_h / T(PH)))
            roi_bin_grid_w = int(np.ceil(roi_w / T(PW)))
        count = T(roi_bin_grid_h * roi_bin_grid_w)
        for c in range(C):
            for ph in range(PH):
                for pw in range(PW):
                    for iy in range(roi_bin_grid_h):
                        y = sh + T(ph) * bin_h + (T(iy) + T(0.5)) * \
                            bin_h / T(roi_bin_grid_h)
                        for ix in range(roi_bin_grid_w):
                            x = sw + T(pw) * bin_w + (T(ix) + T(0.5)) * \
                                bin_w / T(roi_bin_grid_w)
                            w1, w2, w3, w4, x_low, x_high, y_low, y_high = bilinear_interpolate_gradient(
                                H, W, y, x)
                            dtop = top_diff[r, c, ph, pw]
                            if x_low >= 0 and x_high >= 0 and y_low >= 0 and y_high >= 0:
                                bottom_diff[batch_ind, c, y_low,
                                            x_low] += dtop * w1 / count
                                bottom_diff[batch_ind, c, y_low,
                                            x_high] += dtop * w2 / count
                                bottom_diff[batch_ind, c, y_high,
                                            x_low] += dtop * w3 / count
                                bottom_diff[batch_ind, c, y_high,
                                            x_high] += dtop * w4 / count


def test_roi_align_sym():
    dtype = np.float32

    N, C, H, W = 2, 3, 4, 4

    data = np.arange(N * C * H * W).astype(dtype).reshape((N, C, H, W))
    rois = np.array([[0, 1, 1, 3, 3], [1, 2, 2, 3, 3]], dtype=dtype)

    data_sym = mx.sym.Variable('data')
    rois_sym = mx.sym.Variable('rois')

    output_sym = mobula.op.ROIAlign(data=data_sym, rois=rois_sym, pooled_size=(
        2, 2), spatial_scale=1.0, sampling_ratio=1)
    output_sym = mx.sym.MakeLoss(output_sym)

    exe = output_sym.simple_bind(
        ctx=mx.context.current_context(), data=data.shape, rois=rois.shape)
    exe.forward(data=data, rois=rois)

    res = exe.outputs[0].asnumpy()

    exe.backward()
    mx.nd.waitall()


def test_roi_align_nd():
    dtype = np.float32

    N, C, H, W = 2, 3, 4, 4

    data = mx.nd.array(
        np.arange(N * C * H * W).astype(dtype).reshape((N, C, H, W)))
    rois = mx.nd.array(np.array([[0, 1, 1, 3, 3]], dtype=dtype))

    data.attach_grad()
    with mx.autograd.record():
        output = mobula.op.ROIAlign(data=data, rois=rois, pooled_size=(
            2, 2), spatial_scale=1.0, sampling_ratio=1)
    output.backward()
    mx.nd.waitall()


def test_roi_align_value():
    dtype = np.float32

    dlen = 224
    N, C, H, W = 5, 3, 16, 16
    assert H == W
    R = 7
    pooled_size = (3, 4)

    spatial_scale = H * 1.0 / dlen
    sampling_ratio = 0
    data = mx.nd.array(
        np.arange(N * C * W * H).reshape((N, C, H, W)), dtype=dtype)
    # data = mx.nd.random.uniform(0, 1, (N, C, H, W), dtype = dtype)
    center_xy = mx.nd.random.uniform(0, dlen, (R, 2), dtype=dtype)
    wh = mx.nd.random.uniform(0, dlen, (R, 2), dtype=dtype)
    batch_ind = mx.nd.array(np.random.randint(0, N, size=(R, 1)))
    pos = mx.nd.concat(center_xy - wh / 2, center_xy + wh / 2, dim=1)
    rois = mx.nd.concat(batch_ind, pos, dim=1)

    data.attach_grad()
    rois.attach_grad()
    with mx.autograd.record():
        output = mobula.op.ROIAlign(data=data, rois=rois, pooled_size=pooled_size,
                                    spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    dy = mx.nd.random.uniform(-1, 1, (R, C) + pooled_size, dtype=dtype)
    output.backward(dy)
    real_output, [dx, drois] = roialign_forward_backward(data.asnumpy(
    ), rois.asnumpy(), pooled_size, spatial_scale, sampling_ratio, dy.asnumpy())

    bottom_diff = np.zeros(data.shape, dtype=T)
    roialign_backward(bottom_diff, rois.asnumpy(), pooled_size,
                      spatial_scale, sampling_ratio, dy.asnumpy())
    assert_almost_equal(dx, bottom_diff)

    atol = 1e-3
    rtol = 1e-3
    assert_almost_equal(output.asnumpy(), real_output, atol=atol, rtol=rtol)
    assert_almost_equal(data.grad.asnumpy(), dx, atol=atol, rtol=rtol)
    assert_almost_equal(rois.grad.asnumpy(), drois, atol=atol, rtol=rtol)


if __name__ == '__main__':
    test_roi_align_value()
    test_roi_align_sym()
    test_roi_align_nd()
