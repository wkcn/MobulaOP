import mobula_op
from mobula_op.const import req
import functools
import operator

@mobula_op.op.register
class FullyConnected:
    def __init__(self, num_hidden, no_bias = False, flatten = True):
        self.num_hidden = num_hidden
        self.no_bias = no_bias
        self.flatten = flatten
    def forward(self, data, weight, bias = None):
        '''
            data: (batch_size, input_dim) or (x1, x2, ..., xn, input_dim)
            weight: (num_hidden, input_dim)
            bias: (num_hidden, )
            out: (batch_size, num_hidden) or (x1, x2, ..., xn, num_hidden)
            out = data @ weight.T + bias
        '''
        for r in self.req:
            assert r != req.add
        if self.flatten and self.x.ndim != 2:
            x = self.x.reshape((self.x.shape[0], -1)) # (batch_size, input_dim)
        else:
            x = self.x
        self.F.dot(x, weight.T, out = self.y)
        if not self.no_bias:
            bias_shape = [1] * (self.y.ndim - 1) + [self.num_hidden]
            self.y[:] += bias.reshape(bias_shape)
    def backward(self, dy):
        for r in self.req:
            assert r in [req.write, req.inplace]
        weight = self.X[1] 
        if self.flatten:
            dx_reshape = self.dx.reshape((self.dx.shape[0], -1))
            x_reshape = self.x.reshape((self.x.shape[0], -1))
            # dx
            self.F.dot(dy, weight, out = dx_reshape)
            # dw
            self.F.dot(dy.T, x_reshape, out = self.dX[1])
        else:
            # dx
            self.F.dot(dy, weight, out = self.dx)
            # dw
            self.F.dot(dy.reshape((-1, self.num_hidden)).T, self.x.reshape((-1, self.x.shape[-1])), out = self.dX[1])
        # db
        if not self.no_bias:
            self.dX[2][:] = dy.reshape((-1, self.num_hidden)).sum(0).reshape_like(self.dX[2])
    def infer_shape(self, in_shape):
        data_shape, weight_shape = in_shape[:2]
        if not self.no_bias:
            assert len(in_shape) == 3
            bias_shape = in_shape[2]
        else:
            assert len(in_shape) == 2
        if self.flatten:
            batch_size = data_shape[0:1]
            input_dim = functools.reduce(operator.mul, data_shape[1:])
        else:
            batch_size = data_shape[:-1]
            input_dim = data_shape[-1]
        assert len(weight_shape) == 2
        assert weight_shape[0] == self.num_hidden
        assert weight_shape[1] == input_dim
        if not self.no_bias:
            assert len(bias_shape) == 1
            assert bias_shape[0] == self.num_hidden
        out_shape = batch_size + [self.num_hidden]
        return in_shape, [out_shape]
    def list_arguments(self):
        args = ['data', 'weight']
        if not self.no_bias:
            args.append('bias')
        return args
