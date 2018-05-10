import ctypes

class OPLib:
    def __init__(self, lib, func):
        self.lib = lib
        self.func = func
    def __getattr__(self, name):
        cfunc = getattr(self.lib, name)
        def wrapper(*args, **kwargs):
            args_new = [self.func(a) for a in args]
            kwargs_new = dict([(k, self.func(v)) for k, v in kwargs.items()]) 
            print (cfunc, args_new, kwargs_new)
            return cfunc(*args_new, **kwargs_new)
        return wrapper

def load_common_lib(lib_name, func):
    lib = OPLib(ctypes.CDLL(lib_name), func)
    return lib

from .mx_op.load_lib import load_lib as load_lib_mx
def load_lib(lib_name):
    return load_lib_mx(lib_name)
