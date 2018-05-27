dtypes = dict()
try:
    from . import mx
    import mxnet
    dtypes[mxnet.nd.NDArray] = mx
    dtypes[mxnet.sym.Symbol] = mx
except ImportError as e:
    pass

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
