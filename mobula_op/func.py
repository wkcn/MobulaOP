import ctypes
import functools
import os
import sys
import inspect
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
    def __init__(self, name, func):
        self.name = name
        spec = glue.common.getargspec(func)
        assert len(spec.args) == len(spec.defaults), ValueError('Function %s should specify type for each parameter')
        self.par_type = spec.defaults
        self.par_name = spec.args
    def __call__(self, *args, **kwargs):
        def args_gen():
            i = 0
            for a in args:
                yield a
                i += 1
            num_pars = len(self.par_name)
            while i < num_pars:
                yield kwargs[self.par_name[i]]
                i += 1
        # type check
        args_new = []
        backend = None
        dev_id = None
        noncontiguous_list = []
        backend_vars = []
        for a, p in zip(args_gen(), self.par_type):
            if p == IN or p == OUT:
                backend_tmp = glue.backend.get_var_backend(a)
                if backend is not None and backend_tmp != backend:
                    raise ValueError("Don't use multiple backends in a call :-(")
                backend = backend_tmp
                backend_vars.append(a)
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
                ta = backend.convert_type(a, p) if hasattr(backend, 'convert_type') else p(a)
                pa = self.convert_ctype(ta)
            args_new.append(pa)

        assert backend is not None, ValueError("No parameter about backend:-(")

        if hasattr(backend, 'sync_vars'):
            backend.sync_vars(backend_vars)

        if dev_id is not None:
            if func_lib.gpu_lib is None:
                raise RuntimeError("Doesn't support GPU")
            func_lib.gpu_lib.set_device(dev_id)
            rtn = getattr(func_lib.gpu_lib, self.name)(*args_new)
        else:
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

def get_3loop_size(shape, axis):
    # return: outer_size, middle_size, inner_size
    len_shape = len(shape)
    if axis < 0:
        axis += len_shape
    assert 0 <= axis < len_shape
    outer_size = 1
    for i in range(0, axis):
        outer_size *= shape[i]
    inner_size = 1
    for i in range(axis + 1, len_shape):
        inner_size *= shape[i]
    return outer_size, shape[axis], inner_size
