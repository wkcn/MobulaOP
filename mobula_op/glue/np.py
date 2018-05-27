import numpy as np
from .common import *

def get_pointer(v):
    assert v.dtype == np.float32, TypeError('The type of np.ndarray should be float32')
    return v.ctypes.data

def dev_id(a):
    return None

class OpGen(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
        self.cache = dict()
    def __call__(self, *args, **kwargs):
        if self.name not in self.cache:
            # register operator
            self.cache[self.name] = self.register()
        if args[0] == 'np':
            args = args[1:]
        return self.cache[self.name](*args, **kwargs)
    def register(self): 
        def forward(self, *inputs):
            self.in_data = inputs
            self.req = ['write' for _ in range(len(self.in_data))]
            in_shape = get_in_shape(self.in_data)
            out_shape = self.infer_shape(in_shape)[1]
            self.out_data = [self.F.empty(s) for s in out_shape]
            out = self._forward(*inputs)
            if out is not None:
                if type(out) != list:
                    out = [out]
                for i, x in enumerate(out): 
                    self.assign(self.out_data[i], self.req[i], x)
            if len(self.out_data) == 1:
                return self.out_data[0]
            return self.out_data
        def backward(self, out_grad = None, in_data = None, out_data = None, in_grad = None, req = None):

            if in_data is not None:
                self.in_data = in_data
            if out_data is not None:
                self.out_data = out_data

            if in_grad is None:
                in_grad = [self.F.empty_like(d) for d in self.in_data]
            else:
                if type(in_grad) != list:
                    in_grad = [in_grad]
            self.in_grad = in_grad

            if type(out_grad) != list:
                out_grad = [out_grad]
            self.out_grad = out_grad

            if req is None:
                self.req = ['write' for _ in range(len(self.in_data))]
            else:
                assert len(req) == len(self.in_data), ValueError('len(req) should be %d' % len(self.in_data))
                self.req = req
            out = self._backward(*out_grad)
            if out is not None:
                if type(out) != list:
                    out = [out]
                for i in range(op.num_inputs):
                    self.assign(in_grad[i], req[i], out[i])
            if len(in_grad) == 1:
                return in_grad[0]
            return self.in_grad

        np_op_dict = dict(
            __init__ = self.op.__init__,
            __call__ = forward,
            forward = forward,
            backward = backward,
            _forward = self.op.forward,
            _backward = self.op.backward,
            infer_shape = self.op.infer_shape,
            assign = assign,
            F = property(lambda self : np)
        )
        np_op_dict.update(inputs_func)
        np_op = type('_%s_NP_OP' % self.name,
                (self.op, object),
                np_op_dict
        )
        return np_op
