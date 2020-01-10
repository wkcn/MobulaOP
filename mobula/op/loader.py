"""Operator Loader."""
import os
import sys
import re
import time
import ctypes
import json
import warnings
import portalocker
from ..edict import edict
from ..func import CFuncDef, bind, get_func_idcode, get_idcode_hash
from ..build import config, source_to_so_ctx, build_context, file_changed, ENV_PATH
from ..utils import get_git_hash
from ..dtype import DType, TemplateType
from ..version import OP_LOAD_MODULE_BUILD_VERSION
from ..glue.backend import get_glue_modules
from .gen_code import get_gen_rel_code

gen_code = get_gen_rel_code(os.path.dirname(__file__))


if sys.version_info[0] >= 3:
    import importlib.util

    def load_module(name, pathname):
        """Load Module.

        Paramters
        ---------
        name: str
            the name of module.
        pathname:
            the name of path.

        Returns
        -------
        Module
        """
        spec = importlib.util.spec_from_file_location(name, pathname)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
else:
    import imp

    def load_module(name, pathname):
        """Load Module.

        Paramters
        ---------
        name: str
            the name of module.
        pathname:
            the name of path.

        Returns
        -------
        Module
        """
        module = imp.load_source(name, pathname)
        return module


def _get_func_head_reg(name):
    """Get a pattern object for CFunction Head.

    Paramters
    ---------
    name: str
        Function name.

    Returns
    -------
    A pattern object
    """
    return re.compile(r'^\s*{}\s*(.*)'.format(name))


MOBULA_KERNEL_REG = _get_func_head_reg('MOBULA_(KERNEL|FUNC)')

FUNC_REG = re.compile(
    r'^\s*(.*?)\s*\((.*?)\)(?:.*?)*')
CPP_TEMPLATE_REG = re.compile(r'^\s*template\s*\<(.*?)\>\s*')


def _get_template_decl(code):
    match = CPP_TEMPLATE_REG.search(code)
    if match is None:
        return None
    blocks = match.groups()[0].split(',')
    templates = []
    for block in blocks:
        block_sp = block.split()
        dtype, dname = block_sp
        if dtype.strip() == 'typename':
            templates.append(dname.strip())
    return templates


def parse_parameter_decl(decl):
    """Parse the code of parameter declaration

    Parameters
    ----------
    decl : str
        The C++ code of parameter declaration

    Returns
    -------
    Tuple
        (DType Instance,  variable name)
    """
    num_star = decl.count('*')
    assert num_star <= 1,\
        Exception('Only support pass-by-value or pass-by-1-level-pointer, \
            Error declaration: {}'.format(decl))
    is_pointer = num_star > 0
    if is_pointer:
        decl = decl.replace('*', '')
    decl = decl.strip()
    if decl.startswith('const '):
        is_const = True
        decl = decl[len('const '):]
    else:
        is_const = False
    decl_sp = decl.split(' ')

    # type_name and variable_name in C++ code
    type_name, var_name = decl_sp

    # void* func(...)
    if type_name == 'void':
        assert is_pointer
        return DType(ctypes.c_void_p, is_const=is_const), var_name

    # ctype func(...)
    ctype_name = 'c_{}'.format(type_name)
    if hasattr(ctypes, ctype_name):
        ctype = getattr(ctypes, ctype_name)
        if is_pointer:
            ctype = ctypes.POINTER(ctype)
        return DType(ctype, is_const=is_const), var_name

    # template type
    return TemplateType(tname=type_name, is_pointer=is_pointer, is_const=is_const), var_name


def parse_parameters_list(plist):
    """Parse the code of parameter declaration list

    Parameters
    ----------
    plist : str
        The code of parameter declaration list

    Returns
    -------
    rtn_type :
        The type of return value
    func_name : str
        function name
    pars_list: list
        [(DType|TemplateType, variable name), ...]
    """

    match = FUNC_REG.search(plist)
    head, plist = match.groups()
    head_split = re.split(r'\s+', head)
    plist_split = re.split(r'\s*,\s*', plist)
    func_name = head_split[-1]
    rtn_type = head_split[-2] if len(head_split) == 3 else None
    pars_list = list(map(parse_parameter_decl, plist_split))
    return rtn_type, func_name, pars_list


# runtime
CTX_FUNC_MAP = dict()  # ctx -> dict(idcode -> function)
CTX_FUNC_INFO_MAP = dict()  # ctx -> dict(idcode -> func_info)
# static
# fname -> dict([(idcode, template_inst_code), ...])
TEMPLATE_INST_MAP = dict()
# fname -> build_id
TEMPLATE_BUILD_ID_MAP = dict()


class CPPInfo:
    """The class of the C++ file's information.

    Parameters
    ----------
    cpp_fname: str
        the filename of C++ file.
    """

    def __init__(self, cpp_fname):
        self.cpp_fname = cpp_fname
        self.function_args = dict()
        self.dll = None

    def load_dll(self, dll_fname):
        """Load Dynamic-Link Library(*.so or *.dll).

        Parameters
        ----------
        dll_fname:
            The name of Dynamic-Link Library.
        """
        # keep reference
        self.dll = ctypes.CDLL(dll_fname)


def _get_so_prefix(fname):
    path, name = os.path.split(fname)
    return os.path.join(path, 'build', os.path.splitext(name)[0])


def _build_lib(cpp_fname, code_buffer, ctx, target_name):
    cpp_path, cpp_basename = os.path.split(cpp_fname)
    build_path = os.path.join(cpp_path, 'build')
    create_time = time.strftime('%a %Y-%m-%d %H:%M:%S (%z)', time.localtime())
    git_hash = get_git_hash()
    extra_code = gen_code('./templates/header.cpp')(
        cpp_fname=cpp_fname,
        git_hash=git_hash,
        create_time=create_time,
        inc_fname=os.path.join('../..', cpp_basename),
        code=code_buffer)

    build_path_ctx = os.path.join(build_path, ctx)
    os.makedirs(build_path_ctx, exist_ok=True)

    # build so
    cpp_wrapper_fname = os.path.join(build_path_ctx,
                                     os.path.splitext(cpp_basename)[0] + '_wrapper.cpp')
    with open(cpp_wrapper_fname, 'w') as fout:
        fout.write(extra_code)
    # build lib
    srcs = [cpp_wrapper_fname]

    source_to_so_ctx(build_path, srcs, target_name, ctx)


def _dtype_to_tvm_value_type(dtype):
    if dtype.is_pointer:
        return 'v_handle'
    if 'int' in dtype.cname:
        return 'v_int64'
    return 'v_float64'


def _get_args_inst_mx(i, t):
    s = 'args.values[%d].%s' % (i, _dtype_to_tvm_value_type(t))
    if t.is_pointer:
        return '''
          static_cast<{dtype}>(
            static_cast<DLTensor*>({tv})->data)'''.format(dtype=t.cname, tv=s)
    else:
        s = '\n          ' + s
    return s


def _generate_kernel_code(func_idcode_hash, arg_types, arg_names, func_name):
    args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])
    args_inst = ', '.join(arg_names)

    kernel_code = gen_code('./templates/kernel_code.cpp')(
        func_idcode_hash=func_idcode_hash,
        args_def=args_def,
        func_name=func_name,
        args_inst=args_inst)
    kernel_code += '\n'

    args_def_async_mx = ', '.join(['{ctype} {name}'.format(
        ctype='NDArrayHandle' if dtype.is_pointer else dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])

    using_async_mx = all(
        map(lambda dtype: 'void' not in dtype.cname, arg_types))
    if using_async_mx:
        args_inst_mx = [_get_args_inst_mx(i, t)
                        for i, t in enumerate(arg_types)]
        const_loc = []
        for i, dtype in enumerate(arg_types):
            if dtype.is_const and dtype.is_pointer:
                const_loc.append(i)
        num_const = len(const_loc)
        const_loc_code = 'nullptr' if num_const == 0 else 'std::array<int, %d>({%s}).data()' % (
            num_const, ','.join([str(u) for u in const_loc]))
        async_mx_code = gen_code('./templates/async_mx_code.cpp')(
            func_idcode_hash=func_idcode_hash,
            func_name=func_name,
            args_inst=args_inst,
            args_inst_mx=','.join(args_inst_mx),
            num_const=num_const,
            const_loc_code=const_loc_code,
            args_def_async_mx=args_def_async_mx,
        )
        async_mx_code += '\n'
        kernel_code += async_mx_code
    return kernel_code


def _generate_func_code(func_idcode_hash, rtn_type, arg_types, arg_names, func_name):
    if rtn_type is None:
        rtn_type = 'void'

    args_def = ', '.join(['{ctype} {name}'.format(
        ctype=dtype.cname,
        name=name
    ) for dtype, name in zip(arg_types, arg_names)])
    args_inst = ', '.join(arg_names)

    code = '''
MOBULA_DLL %s %s(%s) {
''' % (rtn_type, func_idcode_hash, args_def)
    if rtn_type != 'void':
        code += '  return '
    code += '%s(%s);\n}\n' % (func_name, args_inst)
    return code


def _generate_ordinary_code(cpp_info):
    code_buffer = ''
    # generate ordinary functions code
    for func_name, ord_cfunc in cpp_info.function_args.items():
        if ord_cfunc.template_list:
            continue
        func_idcode = get_func_idcode(func_name, ord_cfunc.arg_types)
        func_idcode_hash = get_idcode_hash(func_idcode)
        func_kind = ord_cfunc.func_kind
        if func_kind == CFuncDef.KERNEL:
            code_buffer += _generate_kernel_code(
                func_idcode_hash, ord_cfunc.arg_types, ord_cfunc.arg_names, '{}_kernel'.format(func_name))
            code_buffer += '\n'
    return code_buffer


def _update_template_inst_map(idcode, tmap, cfunc, arg_types):
    # template function
    func_name = cfunc.func_name
    func_idcode_hash = get_idcode_hash(idcode)
    # Check Template Type Mapping
    template_mapping = dict()
    for rtype, dtype in zip(arg_types, cfunc.arg_types):
        if not isinstance(dtype, TemplateType):
            continue
        tname = dtype.tname
        rtype = str(rtype).replace(
            'const', '').replace('*', '').strip()
        if tname in template_mapping:
            assert template_mapping[tname] == rtype,\
                Exception('Excepted template type {} instead of {}'.
                          format(template_mapping[tname], rtype))
        else:
            template_mapping[tname] = rtype
    assert len(template_mapping) == len(cfunc.template_list),\
        Exception('Template List: {}, mapping: {}'.
                  format(cfunc.template_list, template_mapping))

    template_inst = [template_mapping[tname]
                     for tname in cfunc.template_list]
    template_post = '<%s>' % (', '.join(template_inst))
    rtn_type = cfunc.rtn_type
    if rtn_type in template_mapping:
        rtn_type = template_mapping[rtn_type]

    func_kind = cfunc.func_kind
    if func_kind == CFuncDef.KERNEL:
        code = _generate_kernel_code(func_idcode_hash, arg_types, cfunc.arg_names, '({}_kernel{})'.format(
            func_name, template_post))
    else:
        code = _generate_func_code(
            func_idcode_hash, rtn_type, arg_types, cfunc.arg_names, func_name + template_post)
    tmap[idcode] = code


def _add_function(func_map, func_info_map, func_idcode, cpp_info, dll_fname):
    func_idcode_hash = get_idcode_hash(func_idcode)
    func = getattr(cpp_info.dll, func_idcode_hash, None)
    assert func is not None,\
        Exception('No function `{}` in DLL {}'.format(
            func_idcode, dll_fname))

    old_func = func_map.get(func_idcode, None)
    if old_func is not None:
        if old_func[1] != cpp_info.cpp_fname:
            warnings.warn('The function `{}` in `{}` will be overridden by that in `{}`'.format(
                func_idcode, old_func[1], cpp_info.cpp_fname))

    func_map[func_idcode] = (func, cpp_info.cpp_fname)
    func_info_map[func_idcode] = cpp_info


class OpLoader:
    '''Import Operator Loader.
    It's actual to load the operator.

    Parameters
    ----------
    cfunc: CFuncDef
        The definition of function to call.
    arg_types: list of {DType|TemplateType}
        Argument declaration list.
    ctx: str
        Building context.
    cpp_info: CPPInfo
        Related to cfunc.

    Returns
    -------
    CTX_FUNC_MAP[ctx][idcode] : CFunction
    CTX_FUNC_INFO_MAP[ctx][idcode] : cpp_info
    '''

    def __init__(self, cfunc, arg_types, ctx, cpp_info):
        idcode = get_func_idcode(cfunc.func_name, arg_types)
        if ctx not in CTX_FUNC_MAP:
            CTX_FUNC_MAP[ctx] = dict()
            CTX_FUNC_INFO_MAP[ctx] = dict()
        # func_map: dict mapping idcode to CFunction
        func_map = CTX_FUNC_MAP[ctx]
        func_info_map = CTX_FUNC_INFO_MAP[ctx]

        if idcode not in func_map or func_map[idcode][1] != cpp_info.cpp_fname:
            # load function if idcode is not loaded
            cpp_fname = cpp_info.cpp_fname
            cpp_path, cpp_basename = os.path.split(cpp_fname)
            build_path = os.path.join(cpp_path, 'build')

            use_template = bool(cfunc.template_list)
            os.makedirs(build_path, exist_ok=True)
            build_info_fname = os.path.join(
                build_path, os.path.splitext(cpp_basename)[0] + '.js')
            build_info_fs = open(build_info_fname, 'a+')
            portalocker.lock(build_info_fs, portalocker.LOCK_EX)
            build_info_fs.seek(0)
            js_data = build_info_fs.read()
            if js_data:
                map_data = json.loads(js_data)
            else:
                map_data = dict(version=OP_LOAD_MODULE_BUILD_VERSION)
            del js_data

            # try to load the instance of template function
            # map_data is a dict which records build information
            if map_data.get('version') > OP_LOAD_MODULE_BUILD_VERSION:
                portalocker.unlock(build_info_fs)
                raise Exception(
                    """Unsupported higher version %s of wrapper file (Current MobulaOP ver: %s) :-(.
Please update MobulaOP.""" % (map_data.get('version'), OP_LOAD_MODULE_BUILD_VERSION))
            build_id = map_data.get('build_id', 0)
            is_old_version = map_data.get(
                'version') < OP_LOAD_MODULE_BUILD_VERSION
            # load the information of template functions
            tmap = dict() if is_old_version else map_data.get('functions', dict())
            TEMPLATE_BUILD_ID_MAP[cpp_fname] = build_id
            TEMPLATE_INST_MAP[cpp_fname] = tmap

            so_prefix = _get_so_prefix(cpp_fname)
            # The filename of build target
            dll_fname = '{}_{}_{}.so'.format(so_prefix, ctx, build_id)

            need_to_rebuild = True
            if file_changed(cpp_fname):
                tmap.clear()
            elif os.path.exists(dll_fname):
                # Try to load in template_inst_map
                if not use_template or idcode in tmap:
                    need_to_rebuild = False

            removed_dll_fname = None
            if need_to_rebuild:
                if os.path.exists(dll_fname):
                    # remove old DLL file
                    removed_dll_fname = dll_fname
                    TEMPLATE_BUILD_ID_MAP[cpp_fname] += 1
                    build_id = TEMPLATE_BUILD_ID_MAP[cpp_fname]
                    dll_fname = '{}_{}_{}.so'.format(so_prefix, ctx, build_id)
                # build code
                code_buffer = _generate_ordinary_code(cpp_info)
                if use_template:
                    if idcode not in tmap:
                        _update_template_inst_map(
                            idcode, tmap, cfunc, arg_types)
                    # add template instances code into code_buffer
                    code_buffer += ''.join(tmap.values())

                with build_context():
                    try:
                        _build_lib(cpp_fname, code_buffer, ctx, dll_fname)
                    except:
                        # if build fail, unlock the build info file
                        portalocker.unlock(build_info_fs)
                        raise
                # update tmap
                map_data = dict(version=OP_LOAD_MODULE_BUILD_VERSION,
                                build_id=build_id, functions=tmap)
                # clear the old context and write json data
                build_info_fs.seek(0)
                build_info_fs.truncate()
                json.dump(map_data, build_info_fs)
                build_info_fs.flush()
                os.fsync(build_info_fs.fileno())
            portalocker.unlock(build_info_fs)

            # load all functions in the dll
            cpp_info.load_dll(dll_fname)

            # import all functions
            # ordinary functions
            for func_name, ord_cfunc in cpp_info.function_args.items():
                if not ord_cfunc.template_list:
                    func_idcode = get_func_idcode(
                        func_name, ord_cfunc.arg_types)
                    _add_function(func_map, func_info_map,
                                  func_idcode, cpp_info, dll_fname)

            # template functions
            for func_idcode in tmap.keys():
                _add_function(func_map, func_info_map,
                              func_idcode, cpp_info, dll_fname)

            if removed_dll_fname is not None:
                try:
                    os.remove(removed_dll_fname)
                except Exception:
                    pass

        self.func = func_map[idcode][0]
        self.cpp_info = func_info_map[idcode]
        self.idcode_hash = get_idcode_hash(idcode)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def get_async_func(self, glue_mod):
        async_name = getattr(glue_mod, 'async_name', None)
        if async_name is None:
            return None
        return glue_mod.get_async_func(self.cpp_info, self.idcode_hash)


def _get_functions_from_cpp(cpp_fname):
    unmatched_brackets = 0
    func_def = ''
    func_kind = ''
    func_started = False
    template_list = []
    cpp_info = CPPInfo(cpp_fname=cpp_fname)
    function_args = cpp_info.function_args
    for line in open(cpp_fname):
        if not func_started:
            current_template_list = _get_template_decl(line)
            if current_template_list is not None:
                template_list = current_template_list
            match = MOBULA_KERNEL_REG.search(line)
            if match is not None:
                func_def = ''
                func_kind_str = match.groups()[0]
                if func_kind_str == 'KERNEL':
                    func_kind = CFuncDef.KERNEL
                elif func_kind_str == 'FUNC':
                    func_kind = CFuncDef.FUNC
                else:
                    raise TypeError(
                        'Unknown kind of function: %s' % func_kind_str)
                func_started = True
        # In a declaration of a function
        if func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            func_def += line
            if unmatched_brackets == 0:
                func_def = func_def.replace('\n', '').replace('\r', '')
                func_started = False
                rtn_type, kernel_name, par_list = parse_parameters_list(
                    func_def)
                # template name check
                template_set = set(template_list)
                assert len(template_set) == len(template_list),\
                    Exception('Duplicated template name in {}'.format(
                        ', '.join(template_list)))
                use_template = False
                for dtype, _ in par_list:
                    if isinstance(dtype, TemplateType):
                        assert dtype.tname in template_set,\
                            Exception(
                                "template name '{}' is not defined".format(dtype.tname))
                        use_template = True
                if not use_template:
                    template_list = []

                if func_kind == CFuncDef.KERNEL:
                    assert kernel_name.endswith('_kernel'),\
                        Exception('the postfix of a MOBULA_KERNEL name must be `_kernel`, \
                            e.g. addition_forward_kernel')
                    func_name = kernel_name[:-len('_kernel')]
                elif func_kind == CFuncDef.FUNC:
                    func_name = kernel_name
                else:
                    raise Exception(
                        'Unknown function kind: {}'.format(func_kind))

                # Arguments
                funcdef_args = edict(func_name=func_name,
                                     func_kind=func_kind,
                                     arg_names=[t[1] for t in par_list],
                                     arg_types=[t[0] for t in par_list],
                                     rtn_type=rtn_type,
                                     template_list=template_list,
                                     loader=OpLoader,
                                     loader_kwargs=dict(
                                         cpp_info=cpp_info,
                                     )
                                     )
                template_list = []
                function_args[func_name] = funcdef_args

    assert unmatched_brackets == 0,\
        Exception('# unmatched brackets: {}'.format(unmatched_brackets))

    # Load dynamic file
    functions = dict(
        (name, CFuncDef(**kwargs)) for name, kwargs in function_args.items())
    # Load dynamic function for MXNet
    return functions


def load(module_name, path=''):
    """Load Operator Module

    Parameters
    ----------
    module_name: str
        The name of Operator Module
    path: str
        The path of Operator Module [default = current path]
    """
    op_name = os.path.basename(module_name)
    if not path:
        # Find Operator Module in custom directory first
        custom_path = os.path.join(os.path.dirname(__file__), 'custom')
        if os.path.exists(os.path.join(custom_path, op_name)):
            path = custom_path
    path = os.path.join(path, module_name)

    found = False
    cpp_fname = os.path.join(path, op_name + '.cpp')
    if os.path.exists(cpp_fname):
        found = True
        # Get functions
        functions = _get_functions_from_cpp(cpp_fname)
        bind(functions)

    py_fname = os.path.join(path, op_name + '.py')
    if not os.path.exists(py_fname):
        py_fname = os.path.join(path, '__init__.py')

    if os.path.exists(py_fname):
        found = True
        # Create Operator
        load_module(op_name, py_fname)
    assert found,\
        IOError("{op_name}.cpp or {op_name}.py or __init__.py not found\
 in the path {path}".format(op_name=op_name, path=path))
