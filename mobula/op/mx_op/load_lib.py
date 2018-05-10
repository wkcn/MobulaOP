import mxnet as mx
from mxnet.base import _LIB
import ctypes
from ..load_lib import load_common_lib 

def get_mx_pointer(v):
    cp = ctypes.c_void_p() 
    rtn =  _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp

def mx_func(v):
    if isinstance(v, mx.nd.NDArray):
        return get_mx_pointer(v)
    elif isinstance(v, (int, float)):
        return v
    assert "Unsupported Type: {}".format(type(v))
    return None

def load_lib(lib_name):
    return load_common_lib(lib_name, mx_func)
