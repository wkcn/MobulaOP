import ctypes
import functools
import os

class MobulaFuncLib:
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

class MobulaFunc:
    def __init__(self, name, par_type, func_lib):
        self.name = name
        self.par_type = par_type
        self.func_lib = func_lib
    def __call__(self, *args):
        # type check
        args_new = []
        for a, p in zip(args, self.par_type):
            pa = p(a)
            args_new.append(pa)
        getattr(self.func_lib, self.name)(*args_new)

from .glue.mx import mx_func, dev_id_mx, T
func_lib = MobulaFuncLib(mx_func, dev_id_mx)

def bind(functions):
    for k, v in functions.items():
        assert k not in globals(), "Duplicated function name %s" % k # function overload [todo]
        globals()[k] = MobulaFunc(k, v, func_lib)

functions = dict(
        add = (int, T, T, T),
        roi_align_forward = (int, T, float, int, int, int, int, int, int, T, T),
        roi_align_backward = (int, T, int, float, int, int, int, int, int, int, T, T),
)
bind(functions)
