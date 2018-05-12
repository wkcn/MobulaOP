import ctypes
import functools
import os

def load_common_lib(lib_name, func, dev_id_func = None):
    lib = OPLib(ctypes.CDLL(lib_name), func, dev_id_func)
    return lib

from .mx import mx_func, dev_id_mx
def load_lib(lib_name):
    return load_common_lib(lib_name, mx_func, dev_id_func = dev_id_mx)

class Func:
    def __init__(self, func, dev_id_func):
        self.func = func
        self.dev_id_func = dev_id_func
        lib_path = './mobula/build/mobula_op'
        cpu_lib_fname = "%s_cpu.so" % lib_path
        gpu_lib_fname = "%s_gpu.so" % lib_path
        self.cpu_lib = ctypes.CDLL(cpu_lib_fname)
        self.gpu_lib = ctypes.CDLL(gpu_lib_fname) if os.path.exists(gpu_lib_fname) else None
    def __getattr__(self, name):
        def wrapper(*args):
            args_new = []
            dev_id = None
            for a in args:
                aid = self.dev_id_func(a)
                if aid is not None and dev_id is not None:
                    assert aid == dev_id
                dev_id = aid
                try:
                    args_new.append(self.func(a))
                except TypeError as e:
                    raise TypeError(str(e) + str([type(a) for a in args]))
            if dev_id is not None:
                if self.gpu_lib is None:
                    raise RuntimeError("Doesn't support GPU")
                # gpu
                self.lib.set_device(dev_id)
                cfunc = getattr(self.gpu_lib, name)
                return cfunc(*args_new)
            cfunc = getattr(self.cpu_lib, name)
            return cfunc(*args_new)
        return wrapper

func = Func(mx_func, dev_id_mx)
