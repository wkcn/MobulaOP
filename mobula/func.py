import ctypes
import functools
import os
from . import glue

class MobulaFuncLib:
    def __init__(self):
        lib_path = './mobula/build/mobula_op'
        cpu_lib_fname = "%s_cpu.so" % lib_path
        gpu_lib_fname = "%s_gpu.so" % lib_path
        self.cpu_lib = ctypes.CDLL(cpu_lib_fname)
        self.gpu_lib = ctypes.CDLL(gpu_lib_fname) if os.path.exists(gpu_lib_fname) else None

func_lib = MobulaFuncLib()
T = lambda x : x

class MobulaFunc:
    def __init__(self, name, par_type):
        self.name = name
        self.par_type = par_type
    def __call__(self, *args):
        # type check
        args_new = []
        backend = None
        dev_id = None
        for a, p in zip(args, self.par_type):
            if p == T:
                backend_tmp = glue.backend.get_backend(a)
                if backend is not None and backend_tmp != backend:
                    raise ValueError("Don't use multiple backends in a call :-(")
                backend = backend_tmp
                pa = backend.get_pointer(a)
                aid = backend.dev_id(a)

                if aid is not None and dev_id is not None:
                    raise ValueError("Don't use multiple devices in a call :-(")
                dev_id = aid

            else:
                pa = self.convert_ctype(p(a))
            args_new.append(pa)

        assert backend is not None, ValueError("No parameter about backend:-(")

        if dev_id is not None:
            if func_lib.gpu_lib is None:
                raise RuntimeError("Doesn't support GPU")
            func_lib.gpu_lib.set_device(dev_id)
            return getattr(func_lib.gpu_lib, self.name)(*args_new)
        return getattr(func_lib.cpu_lib, self.name)(*args_new)

    def convert_ctype(self, v):
        if isinstance(v, float):
            return ctypes.c_float(v)
        elif isinstance(v, int):
            return v
        raise TypeError("Unsupported Type: {}".format(type(v)))

def bind(functions):
    for k, v in functions.items():
        assert k not in globals(), "Duplicated function name %s" % k # function overload [todo]
        globals()[k] = MobulaFunc(k, v)

functions = dict(
        add = (int, T, T, T),
        roi_align_forward = (int, T, float, int, int, int, int, int, int, T, T),
        roi_align_backward = (int, T, int, float, int, int, int, int, int, int, T, T),
)
bind(functions)
