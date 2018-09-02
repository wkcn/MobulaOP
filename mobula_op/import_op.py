import os
import sys
import re
import time
import ctypes
from .func import CFuncDef, bind
from .build import config, update_build_path, source_to_so_ctx, build_exit, file_changed, ENV_PATH
from .test_utils import list_gpus, get_git_hash
from .dtype import DType, TemplateType

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
    return TemplateType(is_pointer=is_pointer, is_const=is_const), var_name

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

def get_functions_from_cpp(cpp_fname):
    unmatched_brackets = 0
    func_def = ''
    func_started = False
    templates = None
    functions_args = dict()
    for line in open(cpp_fname):
        if not func_started:
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
                assert kernel_name.endswith('_kernel'), Exception('the postfix of a MOBULA_KERNEL name must be `_kernel`, e.g. addition_forward_kernel')
                func_name = kernel_name[:-len('_kernel')]
                loader = lambda x : print (x)
                # Arguments
                funcdef_args = dict(func_name = func_name,
                        arg_names = [t[1] for t in par_list],
                        arg_types = [t[0] for t in par_list],
                        rtn_type = rtn_type,
                        loader = loader)
                functions_args[func_name] = funcdef_args

    assert unmatched_brackets == 0, Exception('# unmatched brackets: {}'.format(unmatched_brackets))

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
