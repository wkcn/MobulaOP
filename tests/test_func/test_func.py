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
