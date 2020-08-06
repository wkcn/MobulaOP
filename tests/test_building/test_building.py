import ctypes
import os

import mobula
from mobula.testing import assert_almost_equal, gradcheck

# [TODO] change BUILD_PATH


def test_custom_struct():
    class MyStruct(ctypes.Structure):
        _fields_ = [
            ('hello', ctypes.c_int),
            ('mobula', ctypes.c_float),
        ]

    mobula.glue.register_cstruct('MyStruct', MyStruct)
    mobula.op.load('MyStruct', os.path.dirname(__file__))

    res = mobula.func.hello((42, 39))
    assert_almost_equal(res, 42 + 39)


if __name__ == '__main__':
    test_custom_struct()
