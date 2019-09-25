import ctypes
import torch
from .common import *


def get_pointer(v):
    def p(e):
        return ctypes.c_void_p(e.data_ptr())
    if not v.is_contiguous():
        c = v.contiguous()
        return p(c), c
    return p(v)


THDTYPE2CTYPE_MAP = dict()
THDTYPE2CTYPE_MAP[torch.int] = ctypes.c_int
THDTYPE2CTYPE_MAP[torch.float] = ctypes.c_float
THDTYPE2CTYPE_MAP[torch.double] = ctypes.c_double


def get_ctype(v):
    dtype = v.dtype
    ctype = THDTYPE2CTYPE_MAP.get(dtype, None)
    assert ctype is not None, TypeError('Unknown Type: {}'.format(dtype))
    return ctype


def dev_id(a):
    if isinstance(a, torch.Tensor):
        dev = a.device
        return None if dev.type == 'cpu' else dev.index
    return None


class OpGen(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
        self.cache = dict()

    def __call__(self, *args, **kwargs):
        if self.name not in self.cache:
            # register operator
            self.cache[self.name] = self.register()
        inputs, pars = get_in_data(op=self.op, *args, **kwargs)
        return self.cache[self.name](*pars[0], **pars[1])(*inputs)

    def register(self):
        op = self.op
        op_name = self.name

        def get_torch_func(op):
            def forward(ctx, self, *args, **kwargs):
                ctx.self = self
                self.in_data = args
                self.req = ['write' for _ in self.in_data]
                in_shape = get_in_shape(self.in_data)
                out_shape = self.infer_shape(in_shape)[1]
                dtype = self.in_data[0].dtype if self.in_data else torch.float32
                device = self.in_data[0].device if self.in_data else torch.device(
                    'cpu')
                self.out_data = [self.F.empty(
                    s, dtype=dtype, device=device) for s in out_shape]
                out = self._forward(*args, **kwargs)
                if out is not None:
                    if not isinstance(out, (list, tuple)):
                        out = [out]
                    for i, x in enumerate(out):
                        self.assign(self.out_data[i], self.req[i], x)
                if len(self.out_data) == 1:
                    return self.out_data[0]
                return tuple(self.out_data)

            def backward(ctx, *args, **kwargs):
                self = ctx.self
                dtype = self.in_data[0].dtype if self.in_data else torch.float32
                device = self.in_data[0].device if self.in_data else torch.device(
                    'cpu')
                self.in_grad = [self.F.empty_like(d, dtype=dtype, device=device) if d.grad is None
                                else d.grad for d in self.in_data]
                self.out_grad = args
                out = self._backward(*args, **kwargs)
                if out is not None:
                    if not isinstance(out, (list, tuple)):
                        out = [out]
                    num_inputs = len(get_varnames(self._forward))
                    for i in range(num_inputs):
                        self.assign(self.in_grad[i], self.req[i], out[i])

                if len(self.in_grad) == 1:
                    return None, self.in_grad[0]
                return (None, ) + tuple(self.in_grad)

            torch_func_dict = dict(
                forward=staticmethod(forward),
                backward=staticmethod(backward),
            )
            torch_func = type('_%s_TORCH_FUNC' % op_name,
                              (op, torch.autograd.Function),
                              torch_func_dict)
            return torch_func

        def get_torch_nn_module(op, torch_func):
            def __init__(self, *args, **kwargs):
                torch.nn.Module.__init__(self)
                if hasattr(op, '__init__'):
                    op.__init__(self, *args, **kwargs)

            def forward(self, *args, **kwargs):
                return torch_func.apply(self, *args, **kwargs)

            torch_nn_module_dict = dict(
                __init__=__init__,
                forward=forward,
                _forward=op.forward,
                _backward=op.backward,
                assign=assign,
                F=property(lambda self: torch),
            )

            torch_nn_module_dict.update(INPUT_FUNCS)

            torch_nn_module = type('_%s_TORCH_NN_MODULE' % op_name,
                                   (op, torch.nn.Module),
                                   torch_nn_module_dict)
            return torch_nn_module

        torch_func = get_torch_func(op)
        torch_nn_module = get_torch_nn_module(op, torch_func)
        return torch_nn_module


F = torch
