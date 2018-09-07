import os
import sys
import re
import time
import ctypes
import json
from ..func import CFuncDef, bind, get_func_idcode, get_idcode_hash, get_ctype_from_idcode
from ..build import config, update_build_path, source_to_so_ctx, build_context, file_changed, ENV_PATH
from ..test_utils import list_gpus, get_git_hash
from ..dtype import DType, TemplateType
from easydict import EasyDict as edict

def load_module_py2(name, pathname):
    module = imp.load_source(name, pathname)
    return module

def load_module_py3(name, pathname):
    spec = importlib.util.spec_from_file_location(name, pathname)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if sys.version_info[0] >= 3:
    import importlib
    load_module = load_module_py3
else:
    import imp
    load_module = load_module_py2

MOBULA_KERNEL_REG = re.compile(r'^\s*MOBULA_KERNEL.*?')
MOBULA_KERNEL_FUNC_REG = re.compile(r'^\s*MOBULA_KERNEL\s*(.*?)\s*\((.*?)\)(?:.*?)*')
CPP_TEMPLATE_REG = re.compile(r'^\s*template\s*\<(.*?)\>\s*')

def get_template_decl(code):
    u = CPP_TEMPLATE_REG.search(code)
    if u is None:
        return None
    blocks = u.groups()[0].split(',')
    templates = []
    for block in blocks:
        sp = block.split()
        dtype, dname = sp
        if dtype.strip() == 'typename':
            templates.append(dname.strip())
    return templates

def parse_parameter_decl(decl):
    """Parse the code of parameter declaration

    Parameters
    ----------
    decl : str
        The code of parameter declaration

    Returns
    -------
    Tuple
        (DType Instance,  variable name)
    """
    num_star = decl.count('*')
    assert num_star <= 1, Exception('Only support pass-by-value or pass-by-1-level-pointer, Error declaration: {}'.format(decl))
    is_pointer = num_star > 0
    if is_pointer:
        decl = decl.replace('*', '')
    sp = decl.split(' ')
    if sp[0] == 'const':
        is_const = True
        sp = sp[1:]
    else:
        is_const = False
    type_name = sp[0]
    var_name = sp[1]

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

    g = MOBULA_KERNEL_FUNC_REG.search(plist)
    head, plist = g.groups()
    head_split = re.split(r'\s+', head)
    plist_split = re.split(r'\s*,\s*', plist)
    func_name = head_split[-1]
    rtn_type = None
    pars_list = []
    for decl in plist_split:
        dtype, pname = parse_parameter_decl(decl)
        pars_list.append((dtype, pname))
    return rtn_type, func_name, pars_list

# runtime
# if a function has been loaded, it will appear in IMPORTED_FUNC_MAP
IMPORTED_FUNC_MAP = dict() # idcode -> function
# static
TEMPLATE_INST_MAP = dict() # fname -> dict([(idcode, template_inst_code), ...])

def get_template_inst_fname(build_path, name):
    fname = '{}_template.js'.format(name)
    fpath = os.path.join(build_path, fname)
    return fpath

# [discarded] build_index: idcode -> the dynamic library link file
def load_js_map(fname):
    if os.path.exists(fname):
        return json.loads(open(fname).read())
    return dict()

def save_js_map(fname, data):
    with open(fname, 'w') as fout:
        fout.write(json.dumps(data))

# reference: https://stackoverflow.com/questions/50964033/forcing-ctypes-cdll-loadlibrary-to-reload-library-from-file
dlclose_func = ctypes.CDLL(None).dlclose
dlclose_func.argtypes = [ctypes.c_void_p]
dlclose_func.restype = ctypes.c_int

class CPPInfo:
    def __init__(self, cpp_fname):
        self.cpp_fname = cpp_fname
        self.function_args = dict()
        self.dll = None
    def load_dll(self, dll_fname):
        # keep reference
        if self.dll is not None:
            dlclose_func(self.dll._handle)
        self.dll = ctypes.CDLL(dll_fname)

def get_so_path(fname):
    path, name = os.path.split(fname)
    return os.path.join(path, 'build', os.path.splitext(name)[0])

def build_lib(cpp_fname, code_buffer, arch):
    cpp_path, cpp_basename = os.path.split(cpp_fname)
    build_path = os.path.join(cpp_path, 'build')
    create_time = time.strftime('%a %Y-%m-%d %H:%M:%S %Z', time.localtime())
    git_hash = get_git_hash()
    extra_code = '''/*
 * MobulaOP Wrapper generated from the source code %s
 * Created by: MobulaOP %s
 * Create Time: %s
 *
 * WARNING! All changes made in this file will be lost!
 */
#include "%s"
extern "C" {
using namespace mobula;
%s

}''' % (cpp_fname, git_hash, create_time, os.path.join('..', cpp_basename), code_buffer)
    # update_build_path(build_path)
    if not os.path.exists(build_path):
        os.mkdir(build_path)
    # build so for cpu
    cpp_wrapper_fname = os.path.join(build_path, os.path.splitext(cpp_basename)[0] + '_wrapper.cpp')
    with open(cpp_wrapper_fname, 'w') as fout:
        fout.write(extra_code)
    # Build CPU Lib
    srcs = [cpp_wrapper_fname]
    for src in ['defines.cpp', 'context.cpp']:
        srcs.append(os.path.join(ENV_PATH, 'src', src))
    if arch == 'cpu':
        target_name = get_so_path(cpp_fname) + '_cpu.so'
        source_to_so_ctx(build_path, srcs, target_name, 'cpu')
    elif arch == 'cuda':
        # Build GPU Lib
        if len(list_gpus()) > 0:
            target_name = get_so_path(cpp_fname) + '_cuda.so'
            source_to_so_ctx(build_path, srcs, target_name, 'cuda')
    else:
        raise Exception('unsupported architecture: {}'.format(arch))
    return target_name

def op_loader(cfunc, arg_types, arch, cpp_info):
    '''Import Operator Loader
    It's actual to load the operator

    Parameters
    ----------
    cfunc : CFuncDef
        the definition of function to call
    arg_types : list whose element is DType or TemplateType
        argument declaration list
    arch : str
        building architecture
    cpp_info : CPPInfo
        related to cfunc

    Returns
    -------
    IMPORTED_FUNC_MAP[idcode] : CFunction
    '''
    idcode = get_func_idcode(cfunc.func_name, arg_types, arch)
    if idcode not in IMPORTED_FUNC_MAP:
        # load func
        cpp_fname = cpp_info.cpp_fname
        cpp_path, cpp_basename = os.path.split(cpp_fname)
        build_path = os.path.join(cpp_path, 'build')

        dll_fname = get_so_path(cpp_fname) + '_{}.so'.format(arch)
        use_template = len(cfunc.template_list) > 0
        if not os.path.exists(build_path):
            os.makedirs(build_path)
        template_inst_fname = get_template_inst_fname(build_path, os.path.splitext(cpp_basename)[0])

        if cpp_fname not in TEMPLATE_INST_MAP:
            tmap = load_js_map(template_inst_fname)
            TEMPLATE_INST_MAP[cpp_fname] = tmap
        else:
            tmap = TEMPLATE_INST_MAP[cpp_fname]

        need_to_rebuild = True
        if file_changed(cpp_fname):
            need_to_rebuild = True
            tmap.clear()
        else:
            if os.path.exists(dll_fname):
                # Try to load in template_inst_map
                if use_template:
                    if idcode in tmap:
                        need_to_rebuild = False
                else:
                    need_to_rebuild = False

        if need_to_rebuild:
            # build code
            code_buffer = ''
            # generate ordinary functions code
            for func_name, ord_cfunc in cpp_info.function_args.items():
                if len(ord_cfunc.template_list) > 0:
                    continue
                args_def = ', '.join(['{ctype} {name}'.format(
                        ctype=dtype.cname,
                        name=name
                    ) for dtype, name in zip(ord_cfunc.arg_types, ord_cfunc.arg_names)])
                nthread = ord_cfunc.arg_names[0]
                args_inst = ', '.join(ord_cfunc.arg_names)
                code_buffer += '''
void %s(%s) {
    KERNEL_RUN(%s, %s)(%s);
}''' % (func_name, args_def, '{}_kernel'.format(func_name), nthread, args_inst)

            # generate template functions code
            if use_template and idcode not in tmap:
                # template function
                func_name = cfunc.func_name
                tp_idcode = get_func_idcode(func_name, arg_types, arch)
                tp_idcode_hash = get_idcode_hash(tp_idcode)
                # Check Template Type Mapping
                template_mapping = dict()
                for rtype, dtype in zip(arg_types, cfunc.arg_types):
                    if not isinstance(dtype, TemplateType):
                        continue
                    tname = dtype.tname
                    rtype = str(rtype).replace('const','').replace('*','').strip()
                    if tname in template_mapping:
                        assert template_mapping[tname] == rtype, Exception('Excepted template type {} instead of {}'.format(template_mapping[tname], rtype))
                    else:
                        template_mapping[tname] = rtype
                assert len(template_mapping) == len(cfunc.template_list), Exception('Template List: {}, mapping: {}'.format(cfunc.template_list, template_mapping))

                args_def = ', '.join(['{ctype} {name}'.format(
                        ctype=dtype.cname,
                        name=name
                    ) for dtype, name in zip(arg_types, cfunc.arg_names)])

                nthread = cfunc.arg_names[0]
                template_inst = [template_mapping[tname] for tname in cfunc.template_list]
                args_inst = ', '.join(ord_cfunc.arg_names)
                template_post = '<%s>' % (', '.join(template_inst))
                code = '''
void %s(%s) {
    KERNEL_RUN(%s, %s)(%s);
}''' % (tp_idcode_hash, args_def, '({}_kernel{})'.format(func_name, template_post), nthread, args_inst)
                tmap[tp_idcode] = code

            for code in tmap.values():
                code_buffer += code

            with build_context():
                build_lib(cpp_fname, code_buffer, arch)
            # update tmap
            save_js_map(template_inst_fname, tmap)

        # load all functions in the dll
        cpp_info.load_dll(dll_fname)

        # import all functions
        # ordinary functions
        for func_name, ord_cfunc in cpp_info.function_args.items():
            if len(ord_cfunc.template_list) == 0:
                func = getattr(cpp_info.dll, func_name, None)
                assert func is not None, Exception('No function {} in DLL {}'.format(func_name, dll_fname))
                idcode = get_func_idcode(func_name, ord_cfunc.arg_types, arch)
                IMPORTED_FUNC_MAP[idcode] = func

        # template functions
        for tp_idcode in tmap.keys():
            tp_idcode_hash = get_idcode_hash(tp_idcode)
            func = getattr(cpp_info.dll, tp_idcode_hash, None)
            assert func is not None, Exception('No function {} in DLL {}'.format(tp_idcode_hash, dll_fname))
            IMPORTED_FUNC_MAP[tp_idcode] = func

    return IMPORTED_FUNC_MAP[idcode]

def get_functions_from_cpp(cpp_fname):
    unmatched_brackets = 0
    func_def = ''
    func_started = False
    templates = None
    template_list = []
    cpp_info = CPPInfo(cpp_fname=cpp_fname)
    function_args = cpp_info.function_args
    for line in open(cpp_fname):
        if not func_started:
            current_template_list = get_template_decl(line)
            if current_template_list is not None:
                template_list = current_template_list
            u = MOBULA_KERNEL_REG.search(line)
            if u is not None:
                func_def = ''
                func_started = True
        # In a declaration of a function
        if func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            func_def += line
            if unmatched_brackets == 0:
                func_started = False
                templates = None
                rtn_type, kernel_name, par_list = parse_parameters_list(func_def)
                # template name check
                template_set = set(template_list)
                assert len(template_set) == len(template_list), Exception('Duplicated template name in {}'.format(', '.join(template_list)))
                use_template = False
                for dtype, _ in par_list:
                    if isinstance(dtype, TemplateType):
                        assert dtype.tname in template_set, Exception("template name '{}' is not defined".format(dtype.tname))
                        use_template = True
                if not use_template:
                    template_list[:] = []

                assert kernel_name.endswith('_kernel'), Exception('the postfix of a MOBULA_KERNEL name must be `_kernel`, e.g. addition_forward_kernel')
                func_name = kernel_name[:-len('_kernel')]

                # Arguments
                funcdef_args = edict(func_name = func_name,
                        arg_names = [t[1] for t in par_list],
                        arg_types = [t[0] for t in par_list],
                        rtn_type = rtn_type,
                        template_list = template_list,
                        loader = op_loader,
                        loader_kwargs = dict(
                            cpp_info = cpp_info,
                            )
                        )
                template_list[:] = []
                function_args[func_name] = funcdef_args

    assert unmatched_brackets == 0, Exception('# unmatched brackets: {}'.format(unmatched_brackets))

    # Load dynamic file
    functions = dict([(name, CFuncDef(**kwargs)) for name, kwargs in function_args.items()])
    return functions

def load(module_name, path=''):
    path = os.path.join(path, module_name)
    found = False
    op_name = os.path.basename(path)
    cpp_fname = os.path.join(path, op_name + '.cpp')
    if os.path.exists(cpp_fname):
        found = True
        # Get functions
        functions = get_functions_from_cpp(cpp_fname)
        bind(functions)

    py_fname = os.path.join(path, op_name + '.py')
    if not os.path.exists(py_fname):
        py_fname = os.path.join(path, '__init__.py')

    op = None
    if os.path.exists(py_fname):
        found = True
        # Create Operator
        module = load_module(op_name, py_fname)
        op = getattr(module, op_name, None)
    assert found, IOError("{}.cpp or {}.py or __init__.py not found in the path {}".format(op_name, op_name, path))
    return op
