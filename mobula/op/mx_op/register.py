import sys
import functools
import pickle
import base64
import copy
import mxnet as mx
from ..CustomOp import CustomOp

if sys.version_info[0] >= 3:
    pars_encode = lambda x : base64.b64encode(pickle.dumps(x)).decode('utf-8')
    pars_decode = lambda x : pickle.loads(base64.b64decode(x.encode('utf-8')))
else:
    pars_encode = lambda x : pickle.dumps(x)
    pars_decode = lambda x : pickle.loads(x)

def register(op_name):
    if type(op_name) != str:
        op = op_name
        op_name = op.__name__
        return register(op_name)(op)

    def decorator(op):

        def get_mx_op(op):
            input_names = op.forward.__code__.co_varnames[1:]

            def __init__(self, *args, **kwargs):
                mx.operator.CustomOp.__init__(self)
                op.__init__(self, *args, **kwargs)

            def forward(self, is_train, req, in_data, out_data, aux):
                self.in_data = in_data
                self.out_data = out_data
                out = self._forward(*in_data)
                if out is not None:
                    if type(out) != list:
                        out = [out]
                    for i, x in enumerate(out): 
                        self.assign(out_data[i], req[i], x)

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                self.in_grad = in_grad
                self.out_grad = out_grad
                out = self._backward(*out_grad)
                if out is not None:
                    if type(out) != list:
                        out = [out]
                    for i in range(op.num_inputs):
                        self.assign(in_grad[i], req[i], out[i])

            def get_element(data):
                return data[0] if len(data) <= 1 else data

            @property
            def func_X(self):
                return get_element(self.in_data) 
            @property
            def func_Y(self):
                return get_element(self.out_data) 
            @property
            def func_dX(self):
                return get_element(self.in_grad)
            @property
            def func_dY(self):
                return get_element(self.out_grad)

            mx_op = type('_%s_MX_OP' % op_name,
                (mx.operator.CustomOp, op),
                dict(
                    __init__ =  __init__,
                    forward = forward,
                    backward = backward,
                    _forward = op.forward,
                    _backward = op.backward,
                    X = func_X, dX = func_dX,
                    Y = func_Y, dY = func_dY,
                )
            )
            return mx_op

        def get_varnames(func):
            varnames = list(func.__code__.co_varnames[1:])
            return varnames 

        def get_mx_prop(op, mx_op):
            def __init__(self, __pars, *args):
                mx.operator.CustomOpProp.__init__(self)
                self._args, self._kwargs = pars_decode(__pars)

            def create_operator(self, ctx, shapes, dtypes):
                return mx_op(*self._args, **self._kwargs)

            mx_prop = type('_%s_MX_OP_PROP' % op_name,
                (mx.operator.CustomOpProp, op),
                dict(
                    __init__ = __init__,
                    list_arguments = lambda self : get_varnames(op.forward),
                    list_outputs = lambda self : get_varnames(op.backward), 
                    infer_shape = op.infer_shape,
                    create_operator = create_operator
                )
            )
            return mx_prop

        def get_op(*args, **kwargs):
            input_names = get_varnames(op.forward)
            num_inputs = len(input_names)
            op_type = kwargs.pop('op_type')
            if len(args) > num_inputs:
                inputs = args[:num_inputs]
                pars = [args[num_inputs:], kwargs]
            else:
                # len(args) <= num_inputs
                inputs = [None for _ in range(num_inputs)]
                for i, a in enumerate(args):
                    assert input_names[i] not in kwargs
                    inputs[i] = a
                # the rest of parameters
                for i in range(len(args), num_inputs):
                    name = input_names[i]
                    assert name in kwargs
                    inputs[i] = kwargs.pop(name)
                pars = [[], kwargs]
            return mx.nd.Custom(*inputs, __pars = pars_encode(pars), op_type = op_type)

        mx_op = get_mx_op(op)
        mx_prop = get_mx_prop(op, mx_op)
        mx.operator.register(op_name)(mx_prop)

        @functools.wraps(op)
        def wrapper(*args, **kwargs):
            return op(*args, **kwargs) 

        return functools.partial(get_op, op_type = op_name) # wrapper
    return decorator
