import importlib

dtypes = dict()
glues = dict()

def register_backend(glue_name, types_name):
    if not isinstance(types_name, list):
        types_name = [types_name]
    glue = None
    try:
        glue = importlib.import_module('.' + glue_name, __package__)
    except ImportError as e:
        raise e
    if glue is not None:
        for t in types_name:
            sp = t.split('.')
            try:
                e = importlib.import_module(sp[0])
                for s in sp[1:]:
                    e = getattr(e, s)
                dtypes[e] = glue
            except ImportError as e:
                raise e
            glues[glue_name] = glue

# register backends
register_backend('mx', ['mxnet.nd.NDArray', 'mxnet.sym.Symbol'])
register_backend('np', ['numpy.ndarray'])
assert len(dtypes) > 0, RuntimeError("No supported backend :-(")

# create generators cache
for b in dtypes.values():
    b.gen_cache = dict()

def get_var_backend(v):
    return dtypes.get(type(v), None)

def get_args_backend(*args, **kwargs):
    b = None
    def args_gen():
        for a in args:
            yield a
        for a in kwargs.values():
            yield a
    for a in args_gen():
        if type(a) == str:
            t = glues.get(a, None)
        else:
            t = get_var_backend(a)
        if t is not None:
            if b is not None:
                assert b == t, TypeError("Support only 1 backend in a call, now: [%s, %s]" % (str(b), str(t)))
            else:
                b = t
    return b

def op_gen(b, op, name):
    if name not in b.gen_cache:
        b.gen_cache[name] = b.OpGen(op = op, name = name)
    return b.gen_cache[name]