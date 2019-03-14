"""A `Module` implement the `MobulaFunc` class."""
__all__ = ['MobulaFunc', 'bind']


import ctypes
import os
import hashlib
from . import glue
from .dtype import DType, TemplateType, UnknownCType
from .build_utils import config


def get_func_idcode(func_name, arg_types):
    """Get Function IDCode

    Parameters
    ----------
    func_name: str
        the name of function
    arg_types: list of DType

    Returns
    -------
    idcode: str
        IDCode
    """
    arg_types_str = ','.join([e.cname for e in arg_types])
    idcode = '{func_name}:{arg_types_str}'.format(
        func_name=func_name,
        arg_types_str=arg_types_str)
    return idcode


def get_idcode_hash(idcode):
    """Get the hash string of IDCode

    Parameters
    ----------
    idcode: str
    arg_types: list of DType

    Returns
    -------
    Hash String of IDCode: str
    """
    idcode_sp = idcode.split(':')
    func_name = idcode_sp[0]
    md5 = hashlib.md5()
    md5.update(idcode[len(func_name) + 1:].encode('utf-8'))
    return '{}_{}'.format(func_name, md5.hexdigest()[:8])


GPU_CTX_NAME = None
for gpu_ctx in ['cuda', 'hip']:
    gpu_lib_fname = os.path.join(os.path.dirname(__file__), 'build',
                                 'mobula_op_{}.so'.format(gpu_ctx))
    if os.path.exists(gpu_lib_fname):
        GPU_CTX_NAME = gpu_ctx
        break


class CFuncDef:
    """The definition of CFunction."""
    KERNEL = 1
    FUNC = 2

    def __init__(self, func_name, func_kind, arg_names=None, arg_types=None, rtn_type=None,
                 template_list=None, loader=None, loader_kwargs=None):
        if arg_names is None:
            arg_names = list()
        if template_list is None:
            template_list = list()
        self.func_name = func_name
        self.func_kind = func_kind
        self.arg_names = arg_names
        self.arg_types = arg_types
        self.rtn_type = rtn_type
        self.template_list = template_list
        self.loader = loader
        self.loader_kwargs = loader_kwargs

    def __call__(self, arg_datas, arg_types, dev_id, glue_mod=None, using_async=False):
        if dev_id is None:
            ctx = 'cpu'
            dev_id = -1
        else:
            ctx = GPU_CTX_NAME
        # function loader
        func = self.loader(self, arg_types, ctx, **self.loader_kwargs)
        if using_async and glue_mod is not None:
            async_name = getattr(glue_mod, 'async_name', None)
            if async_name is not None:
                return getattr(func, async_name)(*arg_datas)
        if self.func_kind == self.KERNEL:
            return func(dev_id, *arg_datas)
        return func(*arg_datas)


class MobulaFunc:
    """An encapsulation for CFunction

    Parameters:
    -----------
    name: str
        function name
    func: CFuncDef
    """

    def __init__(self, name, func):
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

        glue_mod = self._get_glue_mod(args)
        using_async = config.USING_ASYNC_EXEC and glue_mod is not None and hasattr(
            glue_mod, 'get_async_func')

        if not using_async:
            # Pre-process
            for i in self.wait_to_read_list:
                self._wait_to_read(args[i])
            for i in self.wait_to_write_list:
                self._wait_to_write(args[i])

        for var, ptype in zip(args, self.func.arg_types):
            if ptype.is_pointer:
                # The type of `var` is Tensor.
                data, var_dev_id, ctype = self._get_tensor_info(
                    var, ptype, noncont_var_list, temp_var_list, template_mapping, using_async)
            else:
                # The type of `var` is Scalar.
                data, var_dev_id, ctype = self._get_scalar_info(var, ptype)

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
        for i, vtype in enumerate(arg_types):
            if isinstance(vtype, UnknownCType):
                assert vtype.tname in template_mapping,\
                    Exception('Unknown template name: {}'.format(vtype.tname))
                ctype = template_mapping[vtype.tname]._type_
                arg_types[i] = DType(ctype, vtype.is_const)
                arg_datas[i] = ctype(arg_datas[i])

        rtn = self.func(arg_datas=arg_datas,
                        arg_types=arg_types,
                        dev_id=dev_id,
                        glue_mod=glue_mod,
                        using_async=using_async)

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
    def _get_tensor_info(var, ptype, noncont_var_list, temp_var_list, template_mapping, using_async=False):
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
        using_async: bool
            whether to use asynchronous execution

        Returns
        -------
        data: ctyoes.c_void_p
            the pointer of data
        dev_id: int | None
            the id of device
        ctype: ctypes.POINTER | ctypes.c_*
            the ctype of data
        """

        glue_mod = glue.backend.get_var_glue(var)
        data = glue_mod.get_async_pointer(
            var) if using_async else glue_mod.get_pointer(var)
        if isinstance(data, (list, tuple)):
            # data = (contiguous_array_pointer, contiguous_array_object)
            if ptype.is_const:
                if not using_async:
                    MobulaFunc._wait_to_read(data[1])
                temp_var_list.append(data[1])  # hold a reference
            else:
                if not using_async:
                    MobulaFunc._wait_to_write(data[1])
                noncont_var_list.append((var, data[1]))
            data = data[0]
        dev_id = glue_mod.dev_id(var)
        ctype = ctypes.POINTER(glue_mod.get_ctype(var))
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
    def _get_scalar_info(var, ptype):
        """Get scalar info

        Parameters
        ----------
        var: object
            input variable
        ptype: DType | TemplateType
            the type of argument

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

    @staticmethod
    def _get_glue_mod(datas):
        glue_mod = None
        for var in datas:
            glue_mod_ = glue.backend.get_var_glue(var)
            if glue_mod_ is not None:
                if glue_mod is None:
                    glue_mod = glue_mod_
                else:
                    if glue_mod_ != glue_mod:
                        return None
        return glue_mod

    def build(self, ctx, template_types=None):
        """Build this function

        Parameters
        ----------
        ctx: str
            context Name
        template_types: list or tuple or dict, default: []
            list: a list of template type Names
            tuple: a tuple of template type Names
            dict: a mapping from template name to type name

        Examples
        --------
        >>> mobula.func.add.build('cpu', ['float'])
        """
        arg_types = []
        par_type = self.func.arg_types
        if template_types is None:
            template_types = list()
        if isinstance(template_types, (list, tuple)):
            template_mapping = dict()  # tname -> ctype
            for vtype in par_type:
                if isinstance(vtype, TemplateType):
                    tname = vtype.tname
                    if tname in template_mapping:
                        ctype = template_mapping[tname]
                    else:
                        ctype = getattr(ctypes, 'c_{}'.format(
                            template_types.pop(0)))
                        template_mapping[tname] = ctype
                    arg_types.append(vtype(ctype))
                else:
                    arg_types.append(vtype)
            assert not template_types, Exception('redundant type')
        else:
            assert isinstance(template_types, dict), TypeError(
                'The type of template_types should be list or tuple or dict.')
            template_name = set()
            for vtype in par_type:
                if isinstance(vtype, TemplateType):
                    tname = vtype.tname
                    assert tname in template_types, KeyError(
                        'Unknown Template Type: {}'.format(tname))
                    template_name.add(tname)
                    ctype = getattr(ctypes, 'c_{}'.format(
                        template_types[tname]))
                    arg_types.append(vtype(ctype))
                else:
                    arg_types.append(vtype)
            assert len(template_name) == len(template_types), Exception(
                'Different template name: {} vs {}'.format(
                    template_name, set(template_types.keys())))
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
