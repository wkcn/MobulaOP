"""Backend Manager."""
import importlib

DTYPE_TO_GLUE = dict()  # dtype -> glue_mod
GLUE_NAME_TO_GLUE = dict()  # glue_name -> glue_mod
PKG_NAME_TO_GLUE_ARGS = dict()  # package_name -> (glue_name, types_name)


def _check_glue(glue_mod):
    """Check Glue Module.

    Parameters
    ----------
    glue_mod: module
    """
    func_names = ['get_pointer', 'get_ctype', 'dev_id', 'OpGen']
    for name in func_names:
        assert hasattr(glue_mod, name), AttributeError(
            'Attribute {} not found'.format(name))
    assert hasattr(glue_mod.OpGen, '__call__')
    assert hasattr(glue_mod.OpGen, 'register')


def _register_glue_real(glue_name, types_name):
    global DTYPE_TO_GLUE, GLUE_NAME_TO_GLUE
    if not isinstance(types_name, list):
        types_name = [types_name]
    glue = None
    try:
        glue = importlib.import_module('.' + glue_name, __package__)
    except Exception:
        pass
    if glue is not None:
        for tname in types_name:
            tname_sp = tname.split('.')
            try:
                module = importlib.import_module(tname_sp[0])
                for sub_name in tname_sp[1:]:
                    module = getattr(module, sub_name)
                # create generators cache
                glue.gen_cache = dict()
                DTYPE_TO_GLUE[module] = glue
            except ImportError:
                pass
            _check_glue(glue)
            GLUE_NAME_TO_GLUE[glue_name] = glue


def register_glue(glue_name, types_name):
    """Register a glue module.

    Parameters
    ----------
    glue_name: str
        The name of glue module.
    types_name: list of str
        The list of inputs' class names.
    """
    global PKG_NAME_TO_GLUE_ARGS
    pkg_name = None
    for cls_name in types_name:
        _pkg_name = cls_name.split('.')[0]
        if pkg_name is None:
            pkg_name = _pkg_name
        else:
            assert pkg_name == _pkg_name, TypeError(
                'The name of package should be the same in `types_name`')
    PKG_NAME_TO_GLUE_ARGS[pkg_name] = (glue_name, types_name)


# register glue modules.
register_glue('mx', ['mxnet.nd.NDArray', 'mxnet.sym.Symbol'])
register_glue('np', ['numpy.ndarray'])
register_glue('th', ['torch.Tensor'])


def get_var_type_glue(vtype):
    """Get glue module from variable's type.

    Parameters
    ----------
    vtype: data type

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """
    global DTYPE_TO_GLUE, PKG_NAME_TO_GLUE_ARGS
    glue_mod = DTYPE_TO_GLUE.get(vtype, None)
    if glue_mod is not None:
        return glue_mod
    pkg_name = vtype.__module__.split('.')[0]
    if pkg_name not in PKG_NAME_TO_GLUE_ARGS:
        return None
    # try to register glue_mod
    _register_glue_real(*PKG_NAME_TO_GLUE_ARGS[pkg_name])
    return DTYPE_TO_GLUE.get(vtype, None)


def get_var_glue(var):
    """Get glue module from variable.

    Parameters
    ----------
    var: variable

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """

    return get_var_type_glue(type(var))


def get_args_glue(*args, **kwargs):
    """Get glue module from args and kwargs.

    Parameters
    ----------
    *args
    **kwargs

    Returns
    -------
    Glue Module if glue exists, otherwise None.
    """
    glue_mod = None

    def args_iter():
        for arg in args:
            yield arg
        for arg in kwargs.values():
            yield arg

    for arg in args_iter():
        tmp_glue_mod = get_var_glue(arg)
        if tmp_glue_mod is not None:
            if glue_mod is not None:
                assert glue_mod == tmp_glue_mod,\
                    TypeError('Support only 1 backend in a call, now: [%s, %s]'
                              % (str(glue_mod), str(tmp_glue_mod)))
            else:
                glue_mod = tmp_glue_mod
    return glue_mod


def op_gen(glue_mod, op, name):
    """ Get operator generator of glue module.

    Parameters
    ----------
    glue_mod: Glue Module
    op: object
        The object of custom operator.
    name: str
        The name of custom operator.

    Returns
    -------
    The operator generator of glue module.
    """
    if name not in glue_mod.gen_cache:
        glue_mod.gen_cache[name] = glue_mod.OpGen(op=op, name=name)
    return glue_mod.gen_cache[name]


def get_glue_modules():
    return GLUE_NAME_TO_GLUE.values()
