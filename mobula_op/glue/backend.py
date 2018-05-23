dtypes = dict()
try:
    from . import mx
    import mxnet
    dtypes[mxnet.nd.NDArray] = mx
except ImportError as e:
    pass

def get_backend(v):
    return dtypes[type(v)]

def register(op_name):
    return mx.register(op_name) # tmp
