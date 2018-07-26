import mobula_op
from mobula_op.const import req
import numpy as np

@mobula_op.op.register
class Convolution:
    def __init__(self, kernel, stride = (1, 1), dilate = (1, 1), pad = (0, 0), num_filter = 0, no_bias = False):
        assert len(kernel) == 2
        assert len(stride) == 2
        assert len(dilate) == 2
        assert len(pad) == 2
        assert num_filter > 0
        self.num_filter = num_filter
        self.no_bias = no_bias

        self.kernel_h, self.kernel_w = kernel
        self.pad_h, self.pad_w = pad
        self.dilation_h, self.dilation_w = dilate
        self.stride_h, self.stride_w = stride

    def forward(self, data, weight, bias = None):
        '''
        data: (batch_size, channels, height, width)
        weight: (num_filter, channels, self.kernel_h, self.kernel_w)
        bias: (num_filter, )
        data_col: (channels * self.kernel_h * self.kernel_w, height_col * width_col)
        out: (batch_size, num_filter, height_col, width_col)
        out = weight_reshape @ data_col + bias
        '''
        for r in self.req:
            assert r != req.add 

        batch_size, channels, height, width = data.shape
        height_col, width_col = self.y.shape[2:4]

        # weight_reshape: (num_filter, channels * self.kernel_h * self.kernel_w)
        weight_reshape = weight.reshape((weight.shape[0], -1))
        y_reshape = self.y.reshape((batch_size, self.num_filter, height_col * width_col))

        data_col = self.F.empty((channels * self.kernel_h * self.kernel_w, height_col * width_col), dtype = np.float32)
        for b in range(batch_size):
            mobula_op.func.im2col(data[b], channels, height, width, self.kernel_h, self.kernel_w,
                    self.pad_h, self.pad_w, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w, data_col)
            self.F.dot(weight_reshape, data_col, out = y_reshape[b])

        if not self.no_bias:
            self.y[:] += bias.reshape((1, self.num_filter, 1, 1)) 

    def backward(self, dy):
        for r in self.req:
            assert r in [req.write, req.inplace]
        data, weight = self.X[:2]
        dx, dw = self.dX[:2]

        batch_size, channels, height, width = data.shape
        height_col, width_col = self.y.shape[2:4]

        dy_reshape = dy.reshape((batch_size, self.num_filter, -1))
        dw_reshape = dw.reshape((self.num_filter, -1))
        weight_reshape = weight.reshape((self.num_filter, -1))
        data_col = self.F.empty((channels * self.kernel_h * self.kernel_w, height_col * width_col), dtype = np.float32)
        for b in range(batch_size):
            mobula_op.func.im2col(data[b], channels, height, width, self.kernel_h, self.kernel_w,
                    self.pad_h, self.pad_w, self.stride_h, self.stride_w, self.dilation_h, self.dilation_w, data_col)
            dy_b = dy_reshape[b]
            req_value = mobula_op.const.req.add if b > 0 else mobula_op.const.req.write
            # dw
            mobula_op.math.linalg_gemm(dy_b, data_col, tB = True, out = dw_reshape, req = req_value)
            # dx
            mobula_op.math.linalg_gemm(weight_reshape, dy_b, tA = True, out = data_col)
            mobula_op.func.col2im(data_col, channels, height, width,
                self.kernel_h, self.kernel_w, self.pad_h, self.pad_w, self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w, dx[b])

        if not self.no_bias:
            self.dX[2][:] = dy.sum((0, 2, 3)).reshape_like(self.dX[2][:])

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 4
        batch_size, channels, height, width = in_shape[0]

        height_col = (height + 2 * self.pad_h -
          (self.dilation_h * (self.kernel_h - 1) + 1)) // self.stride_h + 1
        width_col = (width + 2 * self.pad_w -
          (self.dilation_w * (self.kernel_w - 1) + 1)) // self.stride_w + 1

        wshape = in_shape[1]
        assert wshape[0] == self.num_filter
        assert wshape[1] == channels
        assert wshape[2] == self.kernel_h
        assert wshape[3] == self.kernel_w

        if not self.no_bias:
            assert len(in_shape) == 3
            bshape = in_shape[2]
            assert len(bshape) == 1 
            assert bshape[0] == self.num_filter
        else:
            assert len(in_shape) == 2

        return in_shape, [(batch_size, self.num_filter, height_col, width_col)]

    def list_arguments(self):
        args = ['data', 'weight']
        if not self.no_bias:
            args.append('bias')
        return args
