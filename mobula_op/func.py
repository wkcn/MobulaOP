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

class CArray(ctypes.Structure):
    _fields_ = [('size', ctypes.c_int), ('data', ctypes.c_void_p)]

class MobulaFunc:
    TYPE_TO_CTYPE = {int:ctypes.c_int, float:ctypes.c_float, IN:ctypes.c_void_p, OUT:ctypes.c_void_p, None:None}
    def __init__(self, name, func):
        self.name = name
        if isinstance(func, (list, tuple)):
            alias, func = func
            self.name_in_lib = alias
        else:
            self.name_in_lib = name
        spec = glue.common.getargspec(func)
        assert len(spec.args) == len(spec.defaults), ValueError('Function %s should specify type for each parameter')
        self.par_type = spec.defaults
        self.par_name = spec.args
        # register type
        for lib in [func_lib.cpu_lib, func_lib.gpu_lib]:
            if lib is not None:
                libf = getattr(lib, self.name_in_lib)
                libf.restype = self.type_to_ctype(func())
                libf.argtypes = self.types_to_ctypes(self.par_type)
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
        temp_list = []
        backend_vars = []

        def analyze_element(a, p, backend, backend_vars, noncontiguous_list):
            if p == IN or p == OUT:
                backend_vars.append(a)
                pa = backend.get_pointer(a)
                if isinstance(pa, (list, tuple)):
                    if p == OUT:
                        noncontiguous_list.append((a, pa[1]))
                    else: # P == IN
                        temp_list.append(pa[1]) # hold a reference
                    pa = pa[0]
                aid = backend.dev_id(a)

                if aid is not None:
                    if dev_id is not None:
                        assert aid == dev_id, ValueError("Don't use multiple devices in a call :-(")
                    else:
                        dev_id = aid
            else:
                ta = backend.convert_type(a) if hasattr(backend, 'convert_type') else a
                pa = p(ta)
            return pa

        # Pre-Check
        def check_backend(a, backend):
            backend_tmp = glue.backend.get_var_backend(a)
            if backend_tmp is not None and backend is not None and backend_tmp != backend:
                raise ValueError("Don't use multiple backends in a call :-( %s vs %s" % (backend, backend_tmp))
            return backend_tmp

        for a, p in zip(args_gen(), self.par_type):
            if isinstance(p, (list, tuple)):
                for e in a:
                    backend = check_backend(a, backend)
            else:
                backend = check_backend(a, backend)

        extra_pars = [backend, backend_vars, noncontiguous_list]

        for a, p in zip(args_gen(), self.par_type):
            if isinstance(p, (list, tuple)):
                ep = p[0]
                pas = [analyze_element(e, ep, *extra_pars) for e in a]
                if ep == int:
                    ctype = ctypes.c_int
                elif ep == float:
                    ctype = ctypes.c_float
                elif ep in [IN, OUT]:
                    ctype = ctypes.c_void_p
                ca = CArray()
                ca.size = len(pas)
                ca.data = ctypes.cast((ctype * len(pas))(*pas), ctypes.c_void_p)
                args_new.append(ca)
            else:
                pa = analyze_element(a, p, *extra_pars)
                args_new.append(pa)

        if backend is not None:
            assert backend is not None, ValueError("No parameter about backend:-(")

            if hasattr(backend, 'sync_vars'):
                backend.sync_vars(backend_vars)

        if dev_id is not None:
            if func_lib.gpu_lib is None:
                raise RuntimeError("Doesn't support GPU")
            func_lib.gpu_lib.set_device(dev_id)
            rtn = getattr(func_lib.gpu_lib, self.name_in_lib)(*args_new)
        else:
            f = getattr(func_lib.cpu_lib, self.name_in_lib)
            rtn = f(*args_new)
        for source, target in noncontiguous_list:
            source[:] = target
        return rtn

    def type_to_ctype(self, p):
        if isinstance(p, (list, tuple)):
            return CArray
        elif p in MobulaFunc.TYPE_TO_CTYPE:
            return MobulaFunc.TYPE_TO_CTYPE[p]
        else:
            raise TypeError("Unsupported Type: {}".format(p))

    def types_to_ctypes(self, par_types):
        return [self.type_to_ctype(p) for p in par_types]

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
