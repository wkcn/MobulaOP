import mxnet as mx
from mxnet.base import _LIB
import ctypes
import sys
from ..load_lib import load_common_lib 

if sys.version_info[0] >= 3:
    long = int

def get_mx_pointer(v):
    cp = ctypes.c_void_p() 
    rtn =  _LIB.MXNDArrayGetData(v.handle, ctypes.byref(cp))
    return cp

def mx_func(v):
    if isinstance(v, mx.nd.NDArray):
        return get_mx_pointer(v)
    elif isinstance(v, (int, float, long)):
        return v
    raise TypeError("Unsupported Type: {}".format(type(v)))

def dev_id_mx(a):
    if isinstance(a, mx.nd.NDArray):
        return a.context.device_id if a.context.device_type == 'gpu' else None
    return None

def load_lib(lib_name):
    return load_common_lib(lib_name, mx_func, dev_id_func = dev_id_mx)
