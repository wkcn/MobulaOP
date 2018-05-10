import ctypes
import functools

class OPLib:
    def __init__(self, lib, func, dev_id_func = None):
        self.lib = lib
        self.func = func
        self.dev_id_func = dev_id_func
    def __getattr__(self, name):
        if self.dev_id_func is None:
            return self.__getattr_cpu__(name)
        return self.__getattr_gpu__(name)
    def __getattr_cpu__(self, name):
        cfunc = getattr(self.lib, name)
        def wrapper(*args):
            args_new = [self.func(a) for a in args]
            return cfunc(*args_new)
        return wrapper
    def __getattr_gpu__(self, name):
        cfunc = getattr(self.lib, name)
        def wrapper(*args):
            args_new = []
            dev_id = None
            for a in args:
                aid = self.dev_id_func(a)
                if aid is not None and dev_id is not None:
                    assert aid == dev_id
                dev_id = aid
                args_new.append(self.func(a))
            if dev_id is not None:
                self.lib.set_device(dev_id)
            return cfunc(*args_new)
        return wrapper

def load_common_lib(lib_name, func, dev_id_func = None):
    lib = OPLib(ctypes.CDLL(lib_name), func, dev_id_func)
    return lib

from .mx_op.load_lib import load_lib as load_lib_mx
def load_lib(lib_name):
    return load_lib_mx(lib_name)
