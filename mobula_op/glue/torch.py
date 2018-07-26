from .common import *
import torch

def get_pointer(v):
    assert v.dtype == torch.float32, TypeError('The type of torch.Tensor should be float32')
    return v.data_ptr()

def dev_id(a):
    if isinstance(a, torch.Tensor):
        dev = a.device
        return None if dev.type == 'cpu' else dev.index
    return None

def wait_to_read(variables):
    pass

def wait_to_write(variables):
    pass

class OpGen(object):
    def __init__(self, op, name):
        self.op = op
        self.name = name
        self.cache = dict()
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    def register(self):
        raise NotImplementedError

F = torch
