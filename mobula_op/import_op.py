import os
import sys
import re
import time
import ctypes
from .func import CFuncDef, bind
from .build import config, update_build_path, source_to_so_ctx, build_exit, file_changed, ENV_PATH
from .test_utils import list_gpus, get_git_hash
from .dtype import DType

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

class TemplateFuncDLL:
    def __init__(self, ctx, lib):
        self.ctx = ctx
        self.lib = lib
        self.cache = dict()
    def __getattr__(self, name):
        return self.get_func(name)
    def get_func(self, name):
        if name in self.cache:
            return self.cache[name]
        pass

class TemplateFuncLib:
    def __init__(self):
        self.mobula_func_lib = None
        self.cpu_lib = TemplateFuncDLL('cpu', self)
        self.gpu_lib = TemplateFuncDLL('gpu', self)

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
    assert hasattr(ctypes, ctype_name), Exception('Unknown type: {}'.format(type_name))
    ctype = getattr(ctypes, ctype_name)

    if is_pointer:
        ctype = ctypes.POINTER(ctype)
    return DType(ctype, is_const=is_const), var_name

def parse_parameters_list(plist):
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

def get_so_path(fname):
    path, name = os.path.split(fname)
    return os.path.join(path, 'build', os.path.splitext(name)[0])

def build_lib(cpp_fname, code_buffer):

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
    target_name = get_so_path(cpp_fname) + '_cpu.so'

    cpp_wrapper_fname = os.path.join(build_path, os.path.splitext(cpp_basename)[0] + '_wrapper.cpp')

    if not os.path.exists(cpp_wrapper_fname) or file_changed(cpp_fname):
        with open(cpp_wrapper_fname, 'w') as fout:
            fout.write(extra_code)

    # Build CPU Lib
    srcs = [cpp_wrapper_fname]
    for src in ['defines.cpp', 'context.cpp']:
        srcs.append(os.path.join(ENV_PATH, 'src', src))
    source_to_so_ctx(build_path, srcs, target_name, 'cpu')

    # Build GPU Lib
    if len(list_gpus()) > 0:
        target_name = get_so_path(cpp_fname) + '_gpu.so'
        source_to_so_ctx(build_path, srcs, target_name, 'cuda')

def get_functions_from_cpp(cpp_fname):

    unmatched_brackets = 0
    func_def = ''
    func_started = False
    functions_args = dict()
    code_buffer = ''
    for line in open(cpp_fname):
        if not func_started:
            u = MOBULA_KERNEL_REG.search(line)
            if u is not None:
                func_def = ''
                func_started = True
        if func_started:
            unmatched_brackets += line.count('(') - line.count(')')
            func_def += line
            if unmatched_brackets == 0:
                func_started = False
                rtn_type, kernel_name, plist = parse_parameters_list(func_def)
                assert kernel_name.endswith('_kernel'), Exception('the postfix of a MOBULA_KERNEL name must be `_kernel`, e.g. addition_forward_kernel')
                func_name = kernel_name[:-len('_kernel')]
                # Generate function Code
                str_plist = ', '.join(['{} {}'.format(dtype.cname, pname) for dtype, pname in plist])
                str_pname = ', '.join(['{}'.format(pname) for _, pname in plist])
                code_buffer += '''
void %s(%s){
    KERNEL_RUN(%s, %s)(%s);
}''' % (func_name, str_plist, kernel_name, plist[0][1], str_pname)
                # Arguments
                lib_path = get_so_path(cpp_fname)
                cfuncdef_args = dict(func_name = func_name,
                            arg_names = [t[1] for t in plist],
                            arg_types = [t[0] for t in plist],
                            rtn_type = rtn_type,
                            lib_path = lib_path)
                functions_args[func_name] = cfuncdef_args

    assert unmatched_brackets == 0, Exception('# unmatched brackets: {}'.format(unmatched_brackets))

    # Build
    build_lib(cpp_fname, code_buffer)
    build_exit()

    # Load dynamic file
    functions = dict([(name, CFuncDef(**kwargs)) for name, kwargs in functions_args.items()])

    return functions


def import_op(path):
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
