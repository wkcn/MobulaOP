from .. import glue
glue.common.OP_MODULE_GLOBALS = globals()
del glue
from .register import register
from .op_loader import load
