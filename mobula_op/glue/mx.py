import mxnet as mx
from mxnet.base import _LIB
import numpy as np
from .common import *

def get_pointer(v):
    assert v.dtype == np.float32, TypeError('The type of NDArray should be float32')
    cp = ctypes.c_void_p() 
    rtn =  _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp

def dev_id(a):
    if isinstance(a, mx.nd.NDArray):
        return a.context.device_id if a.context.device_type == 'gpu' else None
    return None

class OpGen(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
        self.cache = dict()
    def __call__(self, *args, **kwargs):
        inputs, pars = get_in_data(op = self.op, *args, **kwargs)
        op_type = self.name
        if op_type not in self.cache:
            # register operator
            self.cache[op_type] = True
            self.register()
        if isinstance(inputs[0], mx.nd.NDArray):
            return mx.nd.Custom(*inputs, mobula_pars = pars_encode(pars), op_type = op_type)
        return mx.sym.Custom(*inputs, mobula_pars = pars_encode(pars), op_type = op_type)
    def register(self):
        op = self.op
        op_name = self.name
        def get_mx_op(op):
            def __init__(self, *args, **kwargs):
                mx.operator.CustomOp.__init__(self)
                op.__init__(self, *args, **kwargs)
            def forward(self, is_train, req, in_data, out_data, aux):
                self.in_data = in_data
                self.out_data = out_data
                self.req = req
                out = self._forward(*in_data)
                if out is not None:
                    if type(out) != list:
                        out = [out]
                    for i, x in enumerate(out): 
                        self.assign(out_data[i], req[i], x)
            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.in_grad = in_grad
                self.out_grad = out_grad
                self.req = req
                out = self._backward(*out_grad)
                if out is not None:
                    if type(out) != list:
                        out = [out]
                    for i in range(op.num_inputs):
                        self.assign(in_grad[i], req[i], out[i])
            def get_element(data):
                return data[0] if len(data) <= 1 else data
            def get_zeros_like(self, e):
                return mx.nd.zeros_like(e)
            def get_empty_like(self, e):
                return mx.nd.empty(e.shape)
            mx_op = type('_%s_MX_OP' % op_name,
                (mx.operator.CustomOp, op),
                dict(
                    __init__ =  __init__,
                    forward = forward,
                    backward = backward,
                    _forward = op.forward,
                    _backward = op.backward,
                    X = property(lambda self : self.in_data),
                    Y = property(lambda self : self.out_data),
                    dX = property(lambda self : self.in_grad),
                    dY = property(lambda self : self.out_grad),
                    x = property(lambda self : self.in_data[0]),
                    y = property(lambda self : self.out_data[0]),
                    dx = property(lambda self : self.in_grad[0]),
                    dy = property(lambda self : self.out_grad[0]),
                    get_zeros_like = get_zeros_like,
                    get_empty_like = get_empty_like,
                )
            )
            return mx_op

        def get_mx_prop(op, mx_op):
            def __init__(self, mobula_pars):
                mx.operator.CustomOpProp.__init__(self)
                self._args, self._kwargs = pars_decode(mobula_pars)
                op.__init__(self, *self._args, **self._kwargs)
            def list_outputs(func):
                num_outputs = len(get_varnames(func))
                if num_outputs == 0:
                    return []
                elif num_outputs == 1:
                    return ['output']
                return ['output%d' % i for i in range(num_outputs)]
            def create_operator(self, ctx, shapes, dtypes):
                return mx_op(*self._args, **self._kwargs)
            mx_prop = type('_%s_MX_OP_PROP' % op_name,
                (mx.operator.CustomOpProp, op),
                dict(
                    __init__ = __init__,
                    list_arguments = lambda self : get_varnames(op.forward),
                    list_outputs = lambda self : list_outputs(op.backward),
                    infer_shape = op.infer_shape,
                    create_operator = create_operator
                )
            )
            return mx_prop

        mx_op = get_mx_op(op)
        mx_prop = get_mx_prop(op, mx_op)
        mx.operator.register(op_name)(mx_prop)
