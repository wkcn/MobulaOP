import ctypes
import functools
import os
import sys
import inspect
import hashlib
from . import glue
from .dtype import DType, TemplateType, UnknownCType

def get_func_idcode(func_name, arg_types, arch):
    arg_types_str = ','.join([e.cname for e in arg_types])
    idcode = '{func_name}:{arg_types_str}:{arch}'.format(
            func_name = func_name,
            arg_types_str=arg_types_str,
            arch=arch,
            )
    return idcode

def get_idcode_hash(idcode):
    sp = idcode.split(':')
    func_name = sp[0]
    md5 = hashlib.md5()
    md5.update(idcode[len(func_name)+1:].encode('utf-8'))
    return '{}_{}'.format(func_name, md5.hexdigest()[:8])

def get_ctype_from_idcode(idcode):
    sp = idcode.split(':')
    arg_types_str = sp[1]
    def get_ctype(s):
        s = s.replace('const', '')
        if s.count('*') == 1:
            is_pointer = True
            s = s.replace('*', '')
        else:
            is_pointer = False
        s = s.strip()
        ctype = getattr(ctypes, 'c_{}'.format(s), None)
        assert ctype is not None, TypeError('Wrong IDcode {}'.format(idcode))
        if is_pointer:
            return ctypes.POINTER(ctype)
        return ctype
    return [get_ctype(s) for s in arg_types_str.split(',')]

class CFuncDef:
    def __init__(self, func_name, arg_names=[], arg_types=None, rtn_type=None, template_list=[], loader=None, loader_kwargs=None):
        self.func_name = func_name
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.rtn_type = rtn_type
        self.template_list = template_list
        self.loader = loader
        self.loader_kwargs = loader_kwargs
    def __call__(self, arg_datas, arg_types, dev_id):
        if dev_id is not None:
            set_device(dev_id)
        arch = 'cpu' if dev_id is None else 'cuda'
        # function loader
        func = self.loader(self, arg_types, arch, **self.loader_kwargs)
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
        template_mapping = dict()

        def analyze_element(a, p, backend_inputs, backend_outputs, noncontiguous_list, template_mapping):
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
            assert isinstance(p, (DType, TemplateType)), TypeError('Unknown Data Type: {}'.format(type(p)))
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

                if isinstance(p, DType):
                    expected_ctype = p.ctype
                else:
                    if p.tname in template_mapping:
                        expected_ctype = template_mapping[p.tname]
                    else:
                        template_mapping[p.tname] = expected_ctype = ctype
                assert ctype == expected_ctype, TypeError('Expected Type {} instead of {}'.format(expected_ctype, ctype))
                pa = ctypes.cast(pa, ctype)
            else:
                pa = p.ctype(a)
                dev_id = None
                ctype = p.ctype
            return pa, dev_id, ctype

        extra_pars = [backend_inputs, backend_outputs, noncontiguous_list, template_mapping]

        for a, p in zip(args_gen(), self.par_type):
            assert not isinstance(p, (list, tuple)), Exception('Not supported list or tuple as input variable now')
            pa, aid, ctype = analyze_element(a, p, *extra_pars)
            arg_datas.append(pa)
            if isinstance(ctype, UnknownCType):
                ctype.is_const = p.is_const
                arg_types.append(ctype)
            else:
                arg_types.append(DType(ctype, is_const=p.is_const))

            if aid is not None:
                if dev_id is not None:
                    assert aid == dev_id, ValueError("Don't use multiple devices in a call :-(")
                else:
                    dev_id = aid

        # try to know the unknown ctype
        for i, a in enumerate(arg_types):
            if isinstance(a, UnknownCType):
                assert a.tname in template_mapping, Exception('Unknown template name: {}'.format(tname))
                ctype = template_mapping[a.tname]._type_
                arg_types[i] = DType(ctype, a.is_const)
                arg_datas[i] = ctype(arg_datas[i])

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
