import os
import sys
import re
from .func import IN, OUT, CFuncDef, bind
from .build import config, update_build_path, source_to_so_ctx, build_exit, file_changed, ENV_PATH
from .test_utils import list_gpus

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

def assert_file_exists(fname):
    assert os.path.exists(fname), IOError("{} not found".format(fname))

MOBULA_KERNEL_REG = re.compile(r'^\s*MOBULA_KERNEL.*?')
MOBULA_KERNEL_FUNC_REG = re.compile(r'^\s*MOBULA_KERNEL\s*(.*?)\s*\((.*?)\)(?:.*?)*')

def parse_parameters_list(plist):
    g = MOBULA_KERNEL_FUNC_REG.search(plist)
    head, plist = g.groups()
    head_split = re.split(r'\s+', head)
    plist_split = re.split(r'\s*,\s*', plist)
    func_name = head_split[-1]
    rtn_type = 'void'
    pars_list = []
    for p in plist_split:
        r = re.split(r'\s+', p)
        ptype = ' '.join(r[:-1])
        # remove const
        ptype = re.split(r'\s*const\s*', ptype)[-1]
        pname = r[-1]
        pars_list.append((ptype, pname))
    return rtn_type, func_name, pars_list

def get_so_path(fname):
    path, name = os.path.split(fname)
    return os.path.join(path, 'build', os.path.splitext(name)[0])

def build_lib(cpp_fname, code_buffer):

    cpp_path, cpp_basename = os.path.split(cpp_fname)
    build_path = os.path.join(cpp_path, 'build')

    extra_code = '''
#include "%s"
extern "C" {
using namespace mobula;

%s
}
    ''' % (os.path.join('..', cpp_basename), code_buffer)

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

STR2TYPE = {
    'void': None,
    'int': int,
    'float': float,
    'IN': IN,
    'OUT': OUT
}

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
                # Check Type
                for ptype, pname in plist:
                    assert ptype in STR2TYPE, TypeError('Unsupported Type: {}'.format(ptype))
                # Generate function Code
                str_plist = ', '.join(['{} {}'.format(ptype, pname) for ptype, pname in plist])
                str_pname = ', '.join(['{}'.format(pname) for _, pname in plist])
                code_buffer += '''
void %s(%s){
    KERNEL_RUN(%s, %s)(%s);
}
                ''' % (func_name, str_plist, kernel_name, plist[0][1], str_pname)
                # Arguments
                lib_path = get_so_path(cpp_fname)
                cfuncdef_args = dict(func_name = func_name,
                            arg_names = [t[1] for t in plist],
                            arg_types = [STR2TYPE[t[0]] for t in plist],
                            rtn_type = STR2TYPE[rtn_type],
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
    op_name = os.path.basename(path)
    cpp_fname = os.path.join(path, op_name + '.cpp')
    assert_file_exists(cpp_fname)
    py_fname = os.path.join(path, op_name + '.py')
    assert_file_exists(py_fname)

    # Get functions
    functions = get_functions_from_cpp(cpp_fname)
    bind(functions)

    # Create Operator
    module = load_module(op_name, py_fname)
    op = getattr(module, op_name)
    return op
