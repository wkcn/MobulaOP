import ctypes
from .build_utils import OS_IS_LINUX

CTYPE_INTS = [ctypes.c_short, ctypes.c_int, ctypes.c_long, ctypes.c_longlong]
CTYPE_UINTS = [ctypes.c_ushort, ctypes.c_uint, ctypes.c_ulong, ctypes.c_ulonglong]
def get_ctype_name(ctype):
    # ctype.__name__ = 'c_xxx'
    if ctype in CTYPE_INTS[2:]:
        return 'int{}_t'.format(ctypes.sizeof(ctype) * 8)
    if ctype in CTYPE_UINTS[2:]:
        return 'uint{}_t'.format(ctypes.sizeof(ctype) * 8)
    return ctype.__name__[2:]

class DType:
    _DTYPE_LIST_ = dict()  # () -> inst

    def __init__(self, ctype, is_const=False):
        self.ctype = ctype
        self.is_const = is_const
        self.__reset__()

    def __reset__(self):
        if self.ctype.__name__[:2] == 'LP':
            self.is_pointer = True
            basic_type = self.ctype._type_
        else:
            self.is_pointer = False
            basic_type = self.ctype

        ctype_name = get_ctype_name(basic_type)

        if self.is_const:
            ctype_name = 'const {}'.format(ctype_name)
        if self.is_pointer:
            ctype_name += '*'
        self.cname = ctype_name

    def __repr__(self):
        return self.cname

    def __call__(self, value):
        return self.ctype(value)


class UnknownCType:
    def __init__(self, tname):
        self.tname = tname
        self.is_const = False


class TemplateType:
    def __init__(self, tname, is_pointer, is_const):
        self.tname = tname
        self.is_pointer = is_pointer
        self.is_const = is_const

    def __repr__(self):
        return '<typename {const}{tname}{pointer}>'.format(
            const='const ' if self.is_const else '',
            tname=self.tname,
            pointer='*' if self.is_pointer else '')
