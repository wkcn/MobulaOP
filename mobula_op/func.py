import ctypes
import functools
import os
import sys
from . import glue

class MobulaFuncLib:
    def __init__(self):
        lib_path = os.path.join(os.path.dirname(__file__), 'build/mobula_op')
        cpu_lib_fname = "%s_cpu.so" % lib_path
        gpu_lib_fname = "%s_gpu.so" % lib_path
        self.cpu_lib = ctypes.CDLL(cpu_lib_fname)
        self.gpu_lib = ctypes.CDLL(gpu_lib_fname) if os.path.exists(gpu_lib_fname) else None

func_lib = MobulaFuncLib()
IN = lambda x : x
OUT = lambda x : x

class MobulaFunc:
    def __init__(self, name, par_type):
        self.name = name
        self.par_type = par_type
    def __call__(self, *args):
        # type check
        args_new = []
        backend = None
        dev_id = None
        noncontiguous_list = []
        for a, p in zip(args, self.par_type):
            if p == IN or p == OUT:
                backend_tmp = glue.backend.get_var_backend(a)
                if backend is not None and backend_tmp != backend:
                    raise ValueError("Don't use multiple backends in a call :-(")
                backend = backend_tmp
                pa = backend.get_pointer(a)
                if isinstance(pa, (list, tuple)):
                    if p == OUT:
                        noncontiguous_list.append((a, pa[1]))
                    pa = pa[0]
                aid = backend.dev_id(a)

                if aid is not None:
                    if dev_id is not None:
                        assert aid == dev_id, ValueError("Don't use multiple devices in a call :-(")
                    else:
                        dev_id = aid

            else:
                pa = self.convert_ctype(p(a))
            args_new.append(pa)

        assert backend is not None, ValueError("No parameter about backend:-(")

        if dev_id is not None:
            if func_lib.gpu_lib is None:
                raise RuntimeError("Doesn't support GPU")
            func_lib.gpu_lib.set_device(dev_id)
            rtn = getattr(func_lib.gpu_lib, self.name)(*args_new)
        rtn = getattr(func_lib.cpu_lib, self.name)(*args_new)
        for source, target in noncontiguous_list:
            source[:] = target
        return rtn

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
        add = (int, IN, IN, OUT),
        roi_align_forward = (int, IN, float, int, int, int, int, int, int, IN, OUT),
        roi_align_backward = (int, IN, int, float, int, int, int, int, int, int, OUT, OUT),
)
bind(functions)
