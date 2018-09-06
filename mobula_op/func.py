import ctypes
import functools
import os
import sys
import inspect
from . import glue
from .dtype import DType

def get_func_idcode(func_name, arg_types, arch):
    arg_types_str = ','.join([e.cname for e in arg_types])
    idcode = '{func_name}:{arg_types_str}:{arch}'.format(
            func_name=func_name,
            arg_types_str=arg_types_str,
            arch=arch,
            )
    return idcode

class CFuncDef:
    CFUNC_LIST = dict()
    def __init__(self, func_name, arg_names=[], arg_types=None, rtn_type=None, loader=None, loader_kwargs=None):
        self.func_name = func_name
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.rtn_type = rtn_type
        self.loader = loader
        self.loader_kwargs = loader_kwargs
    def __call__(self, arg_datas, arg_types, dev_id):
        if dev_id is not None:
            set_device(dev_id)
        arch = 'cpu' if dev_id is None else 'cuda'
        idcode = get_func_idcode(self.func_name, arg_types, arch)
        if idcode not in CFuncDef.CFUNC_LIST:
            # function loader
            if self.loader_kwargs is None:
                CFuncDef.CFUNC_LIST[idcode] = self.loader(self, arg_types, arch)
            else:
                CFuncDef.CFUNC_LIST[idcode] = self.loader(self, arg_types, arch, **self.loader_kwargs)
        func = CFuncDef.CFUNC_LIST[idcode]
        return func(*arg_datas)

class MobulaFunc:
    """An encapsulation for CFunction
    """
    def __init__(self, name, func):
        self.name = name
        self.par_name = func.arg_names
        self.par_type = func.arg_types
        self.func = func

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
        arg_datas = []
        dev_id = None
        noncontiguous_list = []
        temp_list = []
        backend_inputs = []
        backend_outputs = []
        arg_types = []

        def analyze_element(a, p, backend_inputs, backend_outputs, noncontiguous_list):
            """Analyze an element

            Parameters
            ----------
            a : variable
            p : data type
            backend_inputs: list
                wait_to_read
            backend_outputs: list
                wait_to_write
            noncontiguous_list : list
                the list of noncontiguous variables
            """
            assert isinstance(p, DType)
            backend = glue.backend.get_var_backend(a)
            if p.is_pointer:
                # multiple-dim array
                if p.is_const:
                    backend_inputs.append(a)
                else:
                    backend_outputs.append(a)

                pa = backend.get_pointer(a)
                if isinstance(pa, (list, tuple)):
                    if p.is_const:
                        temp_list.append(pa[1]) # hold a reference
                    else:
                        noncontiguous_list.append((a, pa[1]))
                    pa = pa[0]
                dev_id = backend.dev_id(a)
                ctype = ctypes.POINTER(backend.get_ctype(a))
                assert ctype == p.ctype, TypeError('Expected Type {} instead of {}'.format(p.ctype, ctype))
                pa = ctypes.cast(pa, ctype)
            else:
                pa = a
                dev_id = None
                ctype = p.ctype
            return pa, dev_id, ctype

        extra_pars = [backend_inputs, backend_outputs, noncontiguous_list]

        for a, p in zip(args_gen(), self.par_type):
            assert not isinstance(p, (list, tuple)), Exception('Not supported list or tuple as input variable now')
            pa, aid, ctype = analyze_element(a, p, *extra_pars)
            arg_datas.append(pa)
            arg_types.append(DType(ctype, is_const=p.is_const))

            if aid is not None:
                if dev_id is not None:
                    assert aid == dev_id, ValueError("Don't use multiple devices in a call :-(")
                else:
                    dev_id = aid

        for var in backend_inputs:
            if hasattr(var, 'wait_to_read'):
                var.wait_to_read()
        for var in backend_outputs:
            if hasattr(var, 'wait_to_write'):
                var.wait_to_write()

        # [TODO] set_device for GPU
        rtn = self.func(arg_datas=arg_datas,
                        arg_types=arg_types,
                        dev_id=dev_id)

        for source, target in noncontiguous_list:
            source[:] = target
        return rtn

def bind(functions):
    for k, func in functions.items():
        assert k not in globals(), "Duplicated function name %s" % k # function overload [todo]
        globals()[k] = MobulaFunc(k, func)
