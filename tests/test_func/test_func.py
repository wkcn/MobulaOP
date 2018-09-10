import mobula
from mobula.test_utils import assert_almost_equal 
import numpy as np
import os
mobula.op.load('./utils', os.path.dirname(__file__))

def test_non_c_contiguous():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    c = np.empty((5, 5))
    s = (slice(None), slice(2,4))
    a_part = a[s]
    b_part = b[s]
    c_part = c[s]
    assert a_part.flags.c_contiguous == False
    assert b_part.flags.c_contiguous == False
    assert c_part.flags.c_contiguous == False
    mobula.func.mul_elemwise(a_part.size, a_part, b_part, c_part)
    assert_almost_equal(a_part * b_part, c_part)
    assert_almost_equal(c[s], c_part)

def test_func_kwargs():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    c = np.empty((5, 5))
    mobula.func.mul_elemwise(n=a.size, a=a, b=b, c=c)
    assert_almost_equal(a*b, c)

def test_default_value_op():
    a = np.random.random((5, 5))
    b = np.random.random((5, 5))
    value = np.random.random((5, 5))
    op = mobula.op.default_add_op[np.ndarray](value=value)
    c = op(a, b)
    assert_almost_equal(a+b, c)
    c = op(a) # a+b[default=value]
    assert_almost_equal(a+value, c)

def test_thread():
    n = 300
    out_1 = np.empty(n // 1)
    out_2 = np.empty(n // 2)
    out_3 = np.empty(n // 3)
    out_4 = np.empty(n * 2)
    out_5 = np.empty(n * 3)
    mobula.func.test_thread(n, out_1, out_2, out_3, out_4, out_5)
    assert_almost_equal(np.arange(n // 1) * 1, out_1)
    assert_almost_equal(np.arange(n // 2) * 2, out_2)
    assert_almost_equal(np.arange(n // 3) * 3, out_3)
    assert_almost_equal(np.arange(n * 2) * 2, out_4)
    assert_almost_equal(np.arange(n * 3) * 3, out_5)
