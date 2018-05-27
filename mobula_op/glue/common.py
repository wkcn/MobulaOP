import sys
import pickle
import inspect
import base64
import ctypes
import functools
from ..op.CustomOp import CustomOp
from . import backend

if sys.version_info[0] >= 3:
    pars_encode = lambda x : base64.b64encode(pickle.dumps(x)).decode('utf-8')
    pars_decode = lambda x : pickle.loads(base64.b64decode(x.encode('utf-8')))
    get_varnames = lambda func : inspect.getfullargspec(func).args[1:]
else:
    pars_encode = lambda x : pickle.dumps(x)
    pars_decode = lambda x : pickle.loads(x)
    get_varnames = lambda func : inspect.getargspec(func).args[1:]

def get_in_data(*args, **kwargs):
    '''
    return:
        inputs: input variances
        pars: parameters of the operator
    '''
    op = kwargs.pop('op')
    input_names = get_varnames(op.forward)
    num_inputs = len(input_names)
    if num_inputs > 0:
        # define input variances in the forward function
        # And the input variances may be in args or kwargs
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
                assert name in kwargs, "Variable %s not found" % name
                inputs[i] = kwargs.pop(name)
            pars = [[], kwargs]
    else:
        # The input variances are in args.
        # And the parameters are in kwargs.
        raise NotImplementedError
    return inputs, pars

def get_in_shape(in_data):
    return [d.shape for d in in_data]

class MobulaOperator(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
    def __call__(self, *args, **kwargs):
        b = backend.get_args_backend(*args, **kwargs)
        return backend.op_gen(b, op = self.op, name = self.name)(*args, **kwargs)

def register(op_name):
    if type(op_name) != str:
        op = op_name
        op_name = op.__name__
        return register(op_name)(op)
    def decorator(op):
        return MobulaOperator(op = op, name = op_name)
    return decorator

inputs_func = dict(
    X = property(lambda self : self.in_data),
    Y = property(lambda self : self.out_data),
    dX = property(lambda self : self.in_grad),
    dY = property(lambda self : self.out_grad),
    x = property(lambda self : self.in_data[0]),
    y = property(lambda self : self.out_data[0]),
    dx = property(lambda self : self.in_grad[0]),
    dy = property(lambda self : self.out_grad[0]),
)
'''
OpGen:
    in_data, out_data, in_grad, out_grad
    req[write/add/null]
    X,Y,dX,dY,x,y,dx,dy
    F
'''
