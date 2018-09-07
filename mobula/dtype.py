import ctypes

class DType:
    _DTYPE_LIST_ = dict() # () -> inst
    def __init__(self, ctype, is_const=False):
        self.ctype = ctype
        self.is_const = is_const
        self.__reset__()
    def __reset__(self):
        name = self.ctype.__name__
        self.is_pointer = False
        if name[:2] == 'LP':
            # pointer
            self.is_pointer = True
            ctype_name = name[5:]
            if ctype_name == 'long':
                ctype_name = 'int64_t'
        else:
            ctype_name = name[2:]
        if self.is_const:
            ctype_name = 'const {}'.format(ctype_name)
        if self.is_pointer:
            ctype_name += '*'
        self.cname = ctype_name
    def __repr__(self):
        return self.cname
    def __call__(self, value):
        return self.ctype(value)

class TemplateType:
    def __init__(self, tname, is_pointer, is_const):
        self.tname = tname
        self.is_pointer = is_pointer
        self.is_const = is_const
