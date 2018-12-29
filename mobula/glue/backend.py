import importlib

DTYPE_TO_GLUE = dict()  # dtype -> glue
GLUE_NAME_TO_GLUE = dict()  # glue_name -> glue
PKG_NAME_TO_GLUE_ARGS = dict()  # package_name -> (glue_name, types_name)


def check_backend(b):
    func_names = ['get_pointer', 'get_ctype', 'dev_id', 'OpGen']
    for name in func_names:
        assert hasattr(b, name), AttributeError(
            'Attribute {} not found'.format(name))
    assert hasattr(b.OpGen, '__call__')
    assert hasattr(b.OpGen, 'register')


def _register_backend_real(glue_name, types_name):
    global DTYPE_TO_GLUE, GLUE_NAME_TO_GLUE
    if not isinstance(types_name, list):
        types_name = [types_name]
    glue = None
    try:
        glue = importlib.import_module('.' + glue_name, __package__)
    except Exception:
        pass
    if glue is not None:
        for t in types_name:
            sp = t.split('.')
            try:
                e = importlib.import_module(sp[0])
                for s in sp[1:]:
                    e = getattr(e, s)
                # create generators cache
                glue.gen_cache = dict()
                DTYPE_TO_GLUE[e] = glue
            except ImportError as e:
                pass
            check_backend(glue)
            GLUE_NAME_TO_GLUE[glue_name] = glue


def register_backend(glue_name, types_name):
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


# register backends
register_backend('mx', ['mxnet.nd.NDArray', 'mxnet.sym.Symbol'])
register_backend('np', ['numpy.ndarray'])
register_backend('th', ['torch.Tensor'])


def get_var_type_backend(v_type):
    global DTYPE_TO_GLUE, PKG_NAME_TO_GLUE_ARGS
    backend = DTYPE_TO_GLUE.get(v_type, None)
    if backend is not None:
        return backend
    pkg_name = v_type.__module__.split('.')[0]
    if pkg_name not in PKG_NAME_TO_GLUE_ARGS:
        return None
    # try to register backend
    _register_backend_real(*PKG_NAME_TO_GLUE_ARGS[pkg_name])
    return DTYPE_TO_GLUE.get(v_type, None)


def get_var_backend(v):
    return get_var_type_backend(type(v))


def get_args_backend(*args, **kwargs):
    b = None

    def args_gen():
        for a in args:
            yield a
        for a in kwargs.values():
            yield a
    for a in args_gen():
        t = get_var_backend(a)
        if t is not None:
            if b is not None:
                assert b == t,\
                    TypeError('Support only 1 backend in a call, now: [%s, %s]'
                              % (str(b), str(t)))
            else:
                b = t
    return b


def op_gen(b, op, name):
    if name not in b.gen_cache:
        b.gen_cache[name] = b.OpGen(op=op, name=name)
    return b.gen_cache[name]
