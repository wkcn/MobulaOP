import sys
import pickle
import inspect
import base64
import ctypes
import functools
from ..op.CustomOp import CustomOp

if sys.version_info[0] >= 3:
    pars_encode = lambda x : base64.b64encode(pickle.dumps(x)).decode('utf-8')
    pars_decode = lambda x : pickle.loads(base64.b64decode(x.encode('utf-8')))
    get_varnames = lambda func : inspect.getfullargspec(func).args[1:]
else:
    pars_encode = lambda x : pickle.dumps(x)
    pars_decode = lambda x : pickle.loads(x)
    get_varnames = lambda func : inspect.getargspec(func).args[1:]

def get_in_data(*args, **kwargs):
    op = kwargs.pop('op')
    input_names = get_varnames(op.forward)
    num_inputs = len(input_names)
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
    return inputs, pars
