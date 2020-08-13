import ctypes
import os

import mobula
from mobula.testing import assert_almost_equal, gradcheck


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


def test_custom_ctensor():
    class CTensor(ctypes.Structure):
        _fields_ = [
            ('data', ctypes.POINTER(ctypes.c_float)),
            ('size', ctypes.c_int),
        ]

    def CTensorConstructor(var):
        glue_mod = mobula.glue.backend.get_var_glue(var)
        tensor = glue_mod.Tensor(var)
        data_ptr = ctypes.cast(tensor.data_ptr, ctypes.POINTER(ctypes.c_float))
        return CTensor(data_ptr, var.size)

    mobula.glue.register_cstruct('CTensor', CTensor, CTensorConstructor)
    mobula.op.load('CTensor', os.path.dirname(__file__))

    import numpy as np
    x = np.array([1, 2, 3], dtype=np.float32)
    y = x + 1
    mobula.func.ctensor_inc(1, x)
    assert_almost_equal(y, x)


def test_build_path():
    new_build_path = os.path.join(os.path.dirname(__file__), 'a_new_path')
    old_build_path = mobula.config.BUILD_PATH
    with mobula.config.TempConfig(BUILD_PATH=new_build_path):
        mobula.config.BUILD_PATH = new_build_path
        module_name = 'BuildPath'
        mobula.op.load(module_name, os.path.dirname(__file__))
        res = mobula.func.TestBuildPath()
        assert res == 42

        def build_existed(path, module_name):
            dirname = os.path.join(path, 'build')
            if not os.path.isdir(dirname):
                return False
            for name in os.listdir(dirname):
                if name.startswith(module_name):
                    return True
            return False

        assert not build_existed(old_build_path, module_name)
        assert build_existed(new_build_path, module_name)


def test_template_build():
    with mobula.config.TempConfig(BUILD_IN_LOCAL_PATH=True):
        mobula.op.load('./test_template', os.path.dirname(__file__))
        mobula.func.mul_elemwise.build('cpu', ['float'])
        mobula.func.mul_elemwise.build('cpu', dict(T='int'))
        assert mobula.config.BUILD_IN_LOCAL_PATH == True
        env_path = os.path.dirname(__file__)
        code_fname = os.path.join(
            env_path,
            'test_template', 'build', 'cpu', 'test_template_wrapper.cpp')
        code = open(code_fname).read()
        '''
        In windows, `ctypes.c_int` is the same as `ctypes.c_long`, whose name is `c_long`. The function of `get_ctype_name` will return `int32_t` :(
        '''
        assert 'mul_elemwise_kernel<float>' in code, code
        assert 'mul_elemwise_kernel<int' in code, code


if __name__ == '__main__':
    test_custom_struct()
    test_custom_ctensor()
    test_build_path()
    test_template_build()
