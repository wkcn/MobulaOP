from . import mx
from .operator import operator

def register_op(op_name):
    return mx.register_op(op_name)
