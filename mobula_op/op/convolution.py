from .register import register
import mobula_op
from mobula_op.const import req
import numpy as np

@register
class Convolution:
    def __init__(self, kernel, stride = (1, 1), dilate = (1, 1), pad = (0, 0), num_filter = 0, no_bias = False):
        assert len(kernel) == 2
        assert len(stride) == 2
        assert len(dilate) == 2
        assert len(pad) == 2
        assert num_filter > 0
        self.kernel = kernel
        self.stride = stride
        self.dilate = dilate
        self.pad = pad
        self.num_filter = num_filter
        self.no_bias = no_bias
    def forward(self, data, weight, bias = None):
        '''
        data: (batch_size, channels, height, width)
        weight: (num_filter, channels, kernel_h, kernel_w)
        bias: (num_filter, )
        data_col: (batch_size, channels * kernel_h * kernel_w, height_col * width_col) 
        out: (batch_size, num_filter, height_col, width_col)
        out = weight_reshape @ data_col + bias
        '''
        for r in self.req:
            assert r != req.add 

        batch_size, channels, height, width = data.shape

        kernel_h, kernel_w = self.kernel
        pad_h, pad_w = self.pad
        dilation_h, dilation_w = self.dilate
        stride_h, stride_w = self.stride

        height_col, width_col = self.y.shape[2:4]
        num_kernels = channels * height_col * width_col

        if not hasattr(self, 'data_col'):
            self.data_col = self.F.empty((batch_size, channels * kernel_h * kernel_w, height_col * width_col), dtype = np.float32)
        mobula_op.func.im2col(data, batch_size * channels, height, width, kernel_h, kernel_w,
                pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, self.data_col)
        # weight_reshape: (num_filter, channels * kernel_h * kernel_w)
        weight_reshape = weight.reshape((weight.shape[0], -1))
        y_reshape = self.y.reshape((self.num_filter, batch_size, height_col * width_col))
        mobula_op.math.dot(weight_reshape, self.data_col, out = y_reshape)
        self.y[:] = self.F.transpose(y_reshape, (1, 0, 2)).reshape((batch_size, self.num_filter, height_col, width_col))

        if not self.no_bias:
            self.y[:] += bias.reshape((1, self.num_filter, 1, 1)) 

    def backward(self, dy):
        for r in self.req:
            assert r in [req.write, req.inplace]
        data, weight = self.X[:2]
        dx, dw = self.dX[:2]

        batch_size, channels, height, width = data.shape

        kernel_h, kernel_w = self.kernel
        pad_h, pad_w = self.pad
        dilation_h, dilation_w = self.dilate
        stride_h, stride_w = self.stride

        height_col, width_col = self.y.shape[2:4]
        num_kernels = channels * height_col * width_col

        # dw
        # data_col_transpose: batch_size * height_col * width_col, channels * kernel_h * kernel_w)
        data_col_transpose = self.data_col.transpose((0, 2, 1)).reshape((-1, channels * kernel_h * kernel_w))
        # dy_transpose: (num_filter, batch_size * height_col * width_col) 
        dy_transpose = dy.transpose((1, 0, 2, 3)).reshape((self.num_filter, -1))
        self.F.dot(dy_transpose, data_col_transpose, out = dw.reshape((self.num_filter, -1)))

        # dx
        # use self.data_col as buffer
        # (batch_size, height_col, width_col, channels, kernel_h, kernel_w)
        data_col_reshape = self.data_col.reshape((batch_size, height_col, width_col, channels, kernel_h, kernel_w))
        mobula_op.math.tensordot(dy, weight, axes = ([1], [0]), out = data_col_reshape)

        # (batch_size, channels * kernel_h * kernel_w, height_col * width_col)
        data_col_transpose = data_col_reshape.reshape((batch_size, height_col * width_col, channels * kernel_h * kernel_w)).transpose((0, 2, 1))

        mobula_op.func.col2im(data_col_transpose, batch_size * channels, height, width,
                kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w, dx)

        if not self.no_bias:
            self.dX[2][:] = dy.sum((0, 2, 3)).reshape_like(self.dX[2][:])

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 4
        batch_size, channels, height, width = in_shape[0]

        kernel_h, kernel_w = self.kernel
        pad_h, pad_w = self.pad
        dilation_h, dilation_w = self.dilate
        stride_h, stride_w = self.stride

        height_col = (height + 2 * pad_h -
          (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
        width_col = (width + 2 * pad_w -
          (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
        num_kernels = channels * height_col * width_col

        wshape = in_shape[1]
        assert wshape[0] == self.num_filter
        assert wshape[1] == channels
        assert wshape[2] == kernel_h 
        assert wshape[3] == kernel_w 

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
