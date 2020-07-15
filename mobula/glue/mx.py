from .common import *
import numpy as np
import mxnet as mx
from mxnet.base import _LIB


if not hasattr(mx.nd.NDArray, 'empty_like'):
    mx.nd.empty_like = lambda x: mx.nd.empty(
        x.shape, dtype=x.dtype, ctx=x.context)
if not hasattr(mx.nd.NDArray, 'wait_to_write'):
    mx.nd.NDArray.wait_to_write = lambda self: _LIB.MXNDArrayWaitToWrite(
        self.handle)
if not hasattr(mx.symbol.Symbol, 'simple_bind'):
    mx.symbol.Symbol.simple_bind = mx.symbol.Symbol._simple_bind


def get_pointer(v):
    cp = ctypes.c_void_p()
    _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp


def get_async_pointer(v):
    return v.handle


def get_ctype(v):
    return NPDTYPE2CTYPE(v.dtype)


def dev_id(a):
    if isinstance(a, mx.nd.NDArray):
        return a.context.device_id if a.context.device_type == 'gpu' else None
    return None


async_name = 'mx'

try:
    MX_LIB_APIS = [_LIB.MXShallowCopyNDArray, _LIB.MXNDArrayFree,
                   _LIB.MXNDArrayGetContext, _LIB.MXNDArrayToDLPack,
                   _LIB.MXEnginePushSyncND]
except AttributeError as e:
    warnings.warn("""Fail to enable asynchronous execution for MXNet, since the version of MXNet is old. It will drop the performance.
In order to improve the performance, please install MXNet whose version >= 1.6.0b20190809""")
    MX_LIB_APIS = None


def get_async_func(cpp_info, func_idcode_hash):
    if MX_LIB_APIS is None:
        return None
    cpp_info.dll.RegisterMXAPI.argtypes = [ctypes.c_void_p] * len(MX_LIB_APIS)
    cpp_info.dll.RegisterMXAPI(*MX_LIB_APIS)
    register_func_for_mx = getattr(
        cpp_info.dll, func_idcode_hash + '_register_mx', None)
    if register_func_for_mx is None:
        return None
    async_func_for_mx = getattr(cpp_info.dll, func_idcode_hash + '_async_mx')
    register_func_for_mx.restype = ctypes.c_void_p
    packed_func_mx = ctypes.c_void_p(register_func_for_mx())
    func = lambda *args: async_func_for_mx(packed_func_mx, *args)
    return func


class OpGen(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
        self.cache = dict()

    def __call__(self, *args, **kwargs):
        inputs, pars = get_in_data(op=self.op, *args, **kwargs)
        op_type = self.name
        name = pars[1].pop('name', None)
        input_type = pars[1].pop('__input_type__', None)
        if input_type is None:
            input_type = type(inputs[0])
        if op_type not in self.cache:
            # register operator
            self.cache[op_type] = True
            self.register()
        if input_type is mx.sym.Symbol:
            return mx.sym.Custom(*inputs, mobula_pars=pars_encode(pars), op_type=op_type)
        if hasattr(mx, 'numpy'):
            inputs = [x.as_nd_ndarray() if isinstance(
                x, mx.np.ndarray) else x for x in inputs]
        return mx.nd.Custom(*inputs, mobula_pars=pars_encode(pars), op_type=op_type, name=name)

    def register(self):
        op = self.op
        op_name = self.name

        def get_mx_op(op):
            def __init__(self, *args, **kwargs):
                self.__mx_prop__ = kwargs.pop('__mx_prop__')
                mx.operator.CustomOp.__init__(self)

            def __getattr__(self, name):
                return self.__dict__.get(name, getattr(self.__mx_prop__, name))

            def forward(self, is_train, req, in_data, out_data, aux):
                self.in_data = in_data
                self.out_data = out_data
                self.req = req
                out = self._forward(*in_data)
                if out is not None:
                    if not isinstance(out, (list, tuple)):
                        out = [out]
                    for i, x in enumerate(out):
                        self.assign(out_data[i], req[i], x)

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.in_grad = in_grad
                self.out_grad = out_grad
                self.req = req
                out = self._backward(*out_grad)
                if out is not None:
                    if not isinstance(out, (list, tuple)):
                        out = [out]
                    num_inputs = len(get_varnames(self._forward))
                    for i in range(num_inputs):
                        self.assign(in_grad[i], req[i], out[i])
            mx_op_dict = dict(
                __init__=__init__,
                __getattr__=__getattr__,
                forward=forward,
                backward=backward,
                _forward=op.forward,
                _backward=op.backward,
                F=property(lambda self: mx.nd),
            )
            mx_op_dict.update(INPUT_FUNCS)
            mx_op = type('_%s_MX_OP' % op_name,
                         (op, mx.operator.CustomOp),
                         mx_op_dict)
            return mx_op

        def get_mx_prop(op, mx_op):
            def __init__(self, mobula_pars):
                self._args, self._kwargs = pars_decode(mobula_pars)
                mx.operator.CustomOpProp.__init__(
                    self, need_top_grad=self._kwargs.pop('need_top_grad', True))
                if hasattr(op, '__init__'):
                    op.__init__(self, *self._args, **self._kwargs)

            def list_outputs(self, func):
                num_outputs = getattr(
                    self, 'num_outputs', len(get_varnames(func)))
                if num_outputs == 0:
                    return []
                if num_outputs == 1:
                    return ['output']
                return ['output%d' % i for i in range(num_outputs)]

            def create_operator(self, ctx, shapes, dtypes):
                with ctx:
                    self._kwargs['__mx_prop__'] = self
                    rtn = mx_op(*self._args, **self._kwargs)
                return rtn

            def infer_type(self, in_type, func):
                num_outputs = getattr(
                    self, 'num_outputs', len(get_varnames(func)))
                dtype = in_type[0] if in_type else np.float32
                return in_type, [dtype] * num_outputs

            mx_prop_dict = dict(
                __init__=__init__,
                list_arguments=lambda self: get_varnames(op.forward),
                list_outputs=lambda self: list_outputs(self, op.backward),
                infer_shape=op.infer_shape,
                create_operator=create_operator,
                infer_type=lambda self, in_type: infer_type(
                    self, in_type, op.backward),
                F=property(lambda self: mx.nd),
            )
            optional_list = ['list_arguments', 'list_outputs', 'infer_type']
            for o in optional_list:
                if hasattr(op, o):
                    mx_prop_dict[o] = getattr(op, o)

            mx_prop = type('_%s_MX_OP_PROP' % op_name,
                           (op, mx.operator.CustomOpProp),
                           mx_prop_dict)
            return mx_prop

        mx_op = get_mx_op(op)
        mx_prop = get_mx_prop(op, mx_op)
        mx.operator.register(op_name)(mx_prop)


F = mx.nd
