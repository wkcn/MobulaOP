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
        inputs, pars = get_in_data(op = self.op, *args, **kwargs)
        if self.name not in self.cache:
            # register operator
            self.cache[self.name] = self.register()
        return self.cache[self.name](*pars[0], **pars[1])(*inputs)
    def register(self): 
        def __call__(self, *inputs): 
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
        np_op_dict = dict(
            __init__ = self.op.__init__,
            __call__ = __call__,
            _forward = self.op.forward,
            _backward = self.op.backward,
            infer_shape = self.op.infer_shape,
            F = property(lambda self : np)
        )
        np_op_dict.update(inputs_func)
        np_op = type('_%s_NP_OP' % self.name,
                (self.op, ),
                np_op_dict
        )
        return np_op
    def assign(self, dst, req, src):
        """Helper function for assigning into dst depending on requirements."""
        if req == 'null':
            return
        elif req == 'write' or req == 'inplace':
            dst[:] = src
        elif req == 'add':
            dst[:] += src
