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
    md5.update(idcode[len(func_name) + 1:].encode('utf-8'))
    return '{}_{}'.format(func_name, md5.hexdigest()[:8])


gpu_ctx_name = None
for gpu_ctx in ['cuda', 'hip']:
    gpu_lib_fname = os.path.join(os.path.dirname(__file__), 'build',
                                 'mobula_op_{}.so'.format(gpu_ctx))
    if os.path.exists(gpu_lib_fname):
        gpu_ctx_name = gpu_ctx
        break


class CFuncDef:
    KERNEL = 1
    FUNC = 2

    def __init__(self, func_name, func_kind, arg_names=[], arg_types=None, rtn_type=None,
                 template_list=[], loader=None, loader_kwargs=None):
        self.func_name = func_name
        self.func_kind = func_kind
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
        if self.func_kind == self.KERNEL:
            return func(dev_id, *arg_datas)
        return func(*arg_datas)


class MobulaFunc:
    """An encapsulation for CFunction
    """

    def __init__(self, name, func):
        """
        Parameters:
        -----------
        name: str
            function name
        func: CFuncDef
        """
        self.name = name
        self.func = func

        self.wait_to_read_list = []
        self.wait_to_write_list = []
        for i, ptype in enumerate(self.func.arg_types):
            if ptype.is_pointer:
                if ptype.is_const:
                    self.wait_to_read_list.append(i)
                else:
                    self.wait_to_write_list.append(i)

    def __call__(self, *args, **kwargs):
        # move kwargs into args
        args = list(args)
        for name in self.func.arg_names[len(args):]:
            args.append(kwargs[name])

        # type check
        arg_datas = []
        dev_id = None
        noncont_var_list = []
        temp_var_list = []
        arg_types = []
        template_mapping = dict()

        # Pre-process
        for i in self.wait_to_read_list:
            self._wait_to_read(args[i])
        for i in self.wait_to_write_list:
            self._wait_to_write(args[i])

        for var, ptype in zip(args, self.func.arg_types):
            if ptype.is_pointer:
                # The type of `var` is Tensor.
                data, var_dev_id, ctype = self._get_tensor_info(
                    var, ptype, noncont_var_list, temp_var_list, template_mapping)
            else:
                # The type of `var` is Scalar.
                data, var_dev_id, ctype = self._get_scalar_info(
                    var, ptype, template_mapping)

            arg_datas.append(data)
            if isinstance(ctype, UnknownCType):
                ctype.is_const = ptype.is_const
                arg_types.append(ctype)
            else:
                arg_types.append(DType(ctype, is_const=ptype.is_const))

            # update `dev_id`
            if var_dev_id is not None:
                if dev_id is not None:
                    assert var_dev_id == dev_id, ValueError(
                        "Don't use multiple devices in a call :-(")
                else:
                    dev_id = var_dev_id

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

        for source, target in noncont_var_list:
            source[:] = target
        return rtn

    @staticmethod
    def _wait_to_read(var):
        if hasattr(var, 'wait_to_read'):
            var.wait_to_read()

    @staticmethod
    def _wait_to_write(var):
        if hasattr(var, 'wait_to_write'):
            var.wait_to_write()

    @staticmethod
    def _get_tensor_info(var, ptype, noncont_var_list, temp_var_list, template_mapping):
        """Get tensor info

        Parameters
        ----------
        var: object
            input variable
        ptype: DType | TemplateType
            the type of argument
        noncont_var_list: list
            the list of noncontiguous variables
        template_mapping: dict
            the mapping from template name to ctype

        Returns
        -------
        data: ctyoes.c_void_p
            the pointer of data
        dev_id: int | None
            the id of device
        ctype: ctypes.POINTER | ctypes.c_*
            the ctype of data
        """

        backend = glue.backend.get_var_backend(var)
        data = backend.get_pointer(var)
        if isinstance(data, (list, tuple)):
            # data = (contiguous_array_pointer, contiguous_array_object)
            if ptype.is_const:
                temp_var_list.append(data[1])  # hold a reference
                MobulaFunc._wait_to_read(data[1])
            else:
                noncont_var_list.append((var, data[1]))
                MobulaFunc._wait_to_write(data[1])
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
        return data, dev_id, ctype

    @staticmethod
    def _get_scalar_info(var, ptype, template_mapping):
        """Get scalar info

        Parameters
        ----------
        var: object
            input variable
        ptype: DType | TemplateType
            the type of argument
        template_mapping: dict
            the mapping from template name to ctype

        Returns
        -------
        data: ctyoes.c_void_p
            the pointer of data
        dev_id: int | None
            the id of device
        ctype: ctypes.POINTER | ctypes.c_*
            the ctype of data
        """

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
        par_type = self.func.arg_types
        if isinstance(template_types, (list, tuple)):
            template_mapping = dict()  # tname -> ctype
            for t in par_type:
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
            for t in par_type:
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
    """Bind Functions to mobula.func.<function name>

    Parameters
    ----------
    functions: dict
        name -> CFuncDef
    """
    for k, func in functions.items():
        assert k not in globals(), "Duplicated function name %s" % k
        globals()[k] = MobulaFunc(k, func)
