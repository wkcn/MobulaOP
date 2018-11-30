import ctypes
import os
import hashlib
from . import glue
from .dtype import DType, TemplateType, UnknownCType


def get_func_idcode(func_name, arg_types):
    arg_types_str = ','.join([e.cname for e in arg_types])
    idcode = '{func_name}:{arg_types_str}'.format(
        func_name=func_name,
        arg_types_str=arg_types_str)
    return idcode


def get_idcode_hash(idcode):
    sp = idcode.split(':')
    func_name = sp[0]
    md5 = hashlib.md5()
    md5.update(idcode[len(func_name)+1:].encode('utf-8'))
    return '{}_{}'.format(func_name, md5.hexdigest()[:8])


gpu_ctx_name = None
for gpu_ctx in ['cuda', 'hip']:
    gpu_lib_fname = os.path.join(os.path.dirname(__file__), 'build',
                                 'mobula_op_{}.so'.format(gpu_ctx))
    if os.path.exists(gpu_lib_fname):
        gpu_ctx_name = gpu_ctx
        break


class CFuncDef:
    def __init__(self, func_name, arg_names=[], arg_types=None, rtn_type=None,
                 template_list=[], loader=None, loader_kwargs=None):
        self.func_name = func_name
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.rtn_type = rtn_type
        self.template_list = template_list
        self.loader = loader
        self.loader_kwargs = loader_kwargs

    def __call__(self, arg_datas, arg_types, dev_id):
        if dev_id is None:
            ctx = 'cpu'
            dev_id = -1
        else:
            ctx = gpu_ctx_name
        # function loader
        func = self.loader(self, arg_types, ctx, **self.loader_kwargs)
        return func(dev_id, *arg_datas)


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
        arg_types = []
        template_mapping = dict()

        def wait_to_read(var):
            if hasattr(var, 'wait_to_read'):
                var.wait_to_read()

        def wait_to_write(var):
            if hasattr(var, 'wait_to_write'):
                var.wait_to_write()

        def _var_wait(var, ptype):
            if ptype.is_pointer:
                if ptype.is_const:
                    # input
                    wait_to_read(var)
                else:
                    # output
                    wait_to_write(var)

        # Pre-process
        for var, ptype in zip(args_gen(), self.par_type):
            _var_wait(var, ptype)

        def analyze_element(var, ptype, noncontiguous_list, template_mapping):
            """Analyze an element

            Parameters
            ----------
            var   : variable
            ptype : data type
            noncontiguous_list : list
                the list of noncontiguous variables
            template_mapping : dict
                the mapping from template name to ctype
            """
            assert isinstance(ptype, (DType, TemplateType)),\
                TypeError('Unknown Data Type: {}'.format(type(ptype)))
            if ptype.is_pointer:
                backend = glue.backend.get_var_backend(var)

                data = backend.get_pointer(var)
                if isinstance(data, (list, tuple)):
                    # data = (contiguous_array_pointer, contiguous_array_object)
                    if ptype.is_const:
                        temp_list.append(data[1])  # hold a reference
                        wait_to_read(data[1])
                    else:
                        noncontiguous_list.append((var, data[1]))
                        wait_to_write(data[1])
                    data = data[0]
                dev_id = backend.dev_id(var)
                ctype = ctypes.POINTER(backend.get_ctype(var))

                if isinstance(ptype, DType):
                    expected_ctype = ptype.ctype
                else:
                    if ptype.tname in template_mapping:
                        expected_ctype = template_mapping[ptype.tname]
                    else:
                        template_mapping[ptype.tname] = expected_ctype = ctype
                assert ctype == expected_ctype,\
                    TypeError('Expected Type {} instead of {}'.format(
                        expected_ctype, ctype))
                data = ctypes.cast(data, ctype)
            else:
                dev_id = None
                if isinstance(ptype, TemplateType):
                    data = var
                    ctype = type(var) if hasattr(
                        var, '_type_') else UnknownCType(ptype.tname)
                else:
                    data = var if isinstance(
                        var, ctypes.c_void_p) else ptype.ctype(var)
                    ctype = ptype.ctype
            return data, dev_id, ctype

        extra_pars = [noncontiguous_list, template_mapping]

        for var, ptype in zip(args_gen(), self.par_type):
            assert not isinstance(ptype, (list, tuple)),\
                Exception('Not supported list or tuple as input variable now')
            data, aid, ctype = analyze_element(var, ptype, *extra_pars)
            arg_datas.append(data)
            if isinstance(ctype, UnknownCType):
                ctype.is_const = ptype.is_const
                arg_types.append(ctype)
            else:
                arg_types.append(DType(ctype, is_const=ptype.is_const))

            if aid is not None:
                if dev_id is not None:
                    assert aid == dev_id, ValueError(
                        "Don't use multiple devices in a call :-(")
                else:
                    dev_id = aid

        # try to know the unknown ctype
        for i, a in enumerate(arg_types):
            if isinstance(a, UnknownCType):
                assert a.tname in template_mapping,\
                    Exception('Unknown template name: {}'.format(a.tname))
                ctype = template_mapping[a.tname]._type_
                arg_types[i] = DType(ctype, a.is_const)
                arg_datas[i] = ctype(arg_datas[i])

        rtn = self.func(arg_datas=arg_datas,
                        arg_types=arg_types,
                        dev_id=dev_id)

        for source, target in noncontiguous_list:
            source[:] = target
        return rtn

    def build(self, ctx, template_types=[]):
        """Build this function

        Parameters
        ----------
        ctx: str
            context Name
        template_types: list or tuple or dict, default: []
            list:
                a list of template type Names
            tuple:
                a tuple of template type Names
            dict:
                a mapping from template name to type name

        Examples:
        mobula.func.add.build('cpu', ['float'])
        """
        arg_types = []
        if isinstance(template_types, (list, tuple)):
            template_mapping = dict()  # tname -> ctype
            for t in self.par_type:
                if isinstance(t, TemplateType):
                    tname = t.tname
                    if tname in template_mapping:
                        ctype = template_mapping[tname]
                    else:
                        ctype = getattr(ctypes, 'c_{}'.format(
                            template_types.pop(0)))
                        template_mapping[tname] = ctype
                    arg_types.append(t(ctype))
                else:
                    arg_types.append(t)
            assert not template_types, Exception('redundant type')
        else:
            assert isinstance(template_types, dict), TypeError(
                'The type of template_types should be list or tuple or dict.')
            template_name = set()
            for t in self.par_type:
                if isinstance(t, TemplateType):
                    tname = t.tname
                    assert tname in template_types, KeyError(
                        'Unknown Template Type: {}'.format(tname))
                    template_name.add(tname)
                    ctype = getattr(ctypes, 'c_{}'.format(
                        template_types[tname]))
                    arg_types.append(t(ctype))
                else:
                    arg_types.append(t)
            assert len(template_name) == len(template_types), Exception(
                'Different template name: {} vs {}'.format(template_name, set(template_types.keys())))
        func = self.func
        func.loader(func, arg_types, ctx, **func.loader_kwargs)


def bind(functions):
    for k, func in functions.items():
        assert k not in globals(), "Duplicated function name %s" % k
        globals()[k] = MobulaFunc(k, func)
