import sys
import pickle
import inspect
import base64
import ctypes
import functools


def pars_encode(x): return base64.b64encode(pickle.dumps(x)).decode('utf-8')


def pars_decode(x): return pickle.loads(base64.b64decode(x.encode('utf-8')))


if sys.version_info[0] >= 3:
    getargspec = inspect.getfullargspec
else:
    getargspec = inspect.getargspec


def get_varnames(func): return getargspec(func).args[1:]


CUSTOM_OP_LIST = dict()
OP_MODULE_GLOBALS = None


def get_in_data(*args, **kwargs):
    '''
    return:
        inputs: input variances
        pars: parameters of the operator
    '''
    op = kwargs.pop('op')
    input_names = get_varnames(op.forward)
    num_inputs = len(input_names)
    defaults = getargspec(op.forward).defaults
    num_defaults = len(defaults) if defaults is not None else 0
    # define input variances in the forward function
    # And the input variances may be in args or kwargs
    if len(args) >= num_inputs:
        inputs = args[:num_inputs]
        pars = [args[num_inputs:], kwargs]
    else:
        # len(args) <= num_inputs
        inputs = [None for _ in range(num_inputs)]
        for i, a in enumerate(args):
            assert input_names[i] not in kwargs
            inputs[i] = a
        # the rest of parameters
        for i in range(len(args), num_inputs - num_defaults):
            name = input_names[i]
            assert name in kwargs, "Variable %s not found" % name
            inputs[i] = kwargs.pop(name)
        num_valid_inputs = num_inputs - num_defaults
        for i in range(num_inputs - num_defaults, num_inputs):
            name = input_names[i]
            if name not in kwargs:
                break
            inputs[i] = kwargs.pop(name)
            num_valid_inputs += 1
        inputs = inputs[:num_valid_inputs]
        pars = [[], kwargs]

    return inputs, pars


def get_in_shape(in_data):
    return [d.shape for d in in_data]


def assign(self, dst, req, src):
    """Helper function for assigning into dst depending on requirements."""
    if req == 'null':
        return
    if req in ('write', 'inplace'):
        dst[:] = src
    elif req == 'add':
        dst[:] += src


backend = None  # wait for importing in __init__.py


class MobulaOperator(object):
    def __init__(self, op, name, **attrs):
        self.op = op
        self.name = name
        self.attrs = attrs

    def __call__(self, *args, **kwargs):
        b = backend.get_args_backend(*args, **kwargs)
        assert b is not None, ValueError('No explict backend')
        new_kwargs = kwargs.copy()
        new_kwargs.update(self.attrs)
        return backend.op_gen(b, op=self.op, name=self.name)(*args, **new_kwargs)

    def __getitem__(self, input_type):
        b = backend.dtypes.get(input_type, None)
        assert b is not None, ValueError(
            'The backend of {} is not found'.format(input_type))

        def wrapper(*args, **kwargs):
            new_kwargs = kwargs.copy()
            new_kwargs.update(self.attrs)
            new_kwargs['__input_type__'] = input_type
            return backend.op_gen(b, op=self.op, name=self.name)(*args, **new_kwargs)
        return wrapper


'''
1. @register
   class XXX
2. @register("OP")
   class XXX
3. @register(a = 3)
   class XXX
'''


def register(op_name=None, **attrs):
    def decorator(op_name, op):
        if op_name is None:
            op_name = op.__name__
        op_inst = MobulaOperator(op=op, name=op_name, **attrs)
        assert op_name not in CUSTOM_OP_LIST,\
            ValueError(
                'Duplicate operator name {}, please rename it'.format(op_name))
        CUSTOM_OP_LIST[op_name] = op_inst
        OP_MODULE_GLOBALS[op_name] = op_inst
        return op_inst
    if op_name is not None and type(op_name) != str:
        return decorator(None, op_name)
    return functools.partial(decorator, op_name)


inputs_func = dict(
    X=property(lambda self: self.in_data),
    Y=property(lambda self: self.out_data),
    dX=property(lambda self: self.in_grad),
    dY=property(lambda self: self.out_grad),
    x=property(lambda self: self.in_data[0]),
    y=property(lambda self: self.out_data[0]),
    dx=property(lambda self: self.in_grad[0]),
    dy=property(lambda self: self.out_grad[0]),
)
'''
OpGen:
    in_data, out_data, in_grad, out_grad
    req[write/add/null]
    X,Y,dX,dY,x,y,dx,dy
    F
'''
try:
    import numpy as np
    NP_DTYPE_NAME2CTYPE = dict()
    pairs = [
        (np.dtype('int8'), ctypes.c_int8),
        (np.dtype('int16'), ctypes.c_int16),
        (np.dtype('int32'), ctypes.c_int32),
        (np.dtype('int64'), ctypes.c_int64),  # alias: np.int
        (np.dtype('float32'), ctypes.c_float),
        (np.dtype('float64'), ctypes.c_double),  # alias: np.float
    ]
    for dtype, ctype in pairs:
        NP_DTYPE_NAME2CTYPE[dtype.name] = ctype

    def NPDTYPE2CTYPE(dtype):
        ctype = NP_DTYPE_NAME2CTYPE.get(np.dtype(dtype).name, None)
        assert ctype is not None, TypeError('Unknown Type: {}'.format(dtype))
        return ctype
except ImportError:
    pass
