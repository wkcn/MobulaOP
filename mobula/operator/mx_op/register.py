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
    def decorator(op):
        class_func = ['__init__', 'forward', 'backward', 'infer_shape']
        for func_name in class_func:
            f = op.__dict__.get(func_name, CustomOp.__dict__[func_name])
            setattr(op, func_name, classmethod(f))

        def get_mx_op(op):
            input_names = op.forward.__code__.co_varnames[1:]
            __init__ = op.__init__

            def __init__(self, *args, **kwargs):
                mx.operator.CustomOp.__init__(self)
                op.__init__(*args, **kwargs)

            def forward(self, is_train, req, in_data, out_data, aux):
                '''
                self.Xs = in_data
                self.X = in_data[0]
                self.Ys = out_data
                self.Y = out_data[0]
                '''
                out = op.forward(*in_data)
                if type(out) != list:
                    out = [out]
                for i, x in enumerate(out): 
                    self.assign(out_data[i], req[i], x)

            def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
                out = op.backward(self, *out_grad)
                if type(out) != list:
                    out = [out]
                for i in range(op.num_inputs):
                    self.assign(in_grad[i], req[i], out[i])

            mx_op = type('_%s_MX_OP' % op_name,
                (mx.operator.CustomOp,),
                dict(
                    __init__ =  __init__,
                    forward = forward,
                    backward = backward
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
                (mx.operator.CustomOpProp,),
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
