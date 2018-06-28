import mxnet as mx
import numpy as np
import mobula_op
from mobula_op.test_utils import assert_almost_equal

def test_tensordot():
    a = np.arange(60.).reshape(3,4,5).astype(np.float32)
    b = np.arange(24.).reshape(4,3,2).astype(np.float32)
    axes = ([1, 0], [0, 1])
    c = np.tensordot(a, b, axes = axes)
    d = mobula_op.math.tensordot(a, b, axes = axes) 
    assert_almost_equal(c, d)

def check_math_func(func, target, **kwargs):
    out = func(**kwargs)
    assert_almost_equal(out, target)
    out = np.zeros_like(out, dtype = np.float32)
    func(out = out, **kwargs)
    assert_almost_equal(out, target)
    # test req add
    base = np.random.random(out.shape).astype(np.float32)
    out = base.copy()
    func(out = out, req = mobula_op.const.req.add, **kwargs)
    assert_almost_equal(out, target + base)
    func(out = out, req = mobula_op.const.req.write, **kwargs)
    assert_almost_equal(out, target)

def test_linalg_gemm():
    I, J, K = 10, 11, 12
    a = np.random.random((I, J)).astype(np.float32)
    b = np.random.random((J, K)).astype(np.float32)
    c = np.empty((I, K), dtype = np.float32)
    t = np.dot(a, b)
    check_math_func(mobula_op.math.linalg_gemm, t, a = a, b = b, tA = False, tB = False)

    b = b.reshape((K, J))
    t = np.dot(a, b.T)
    check_math_func(mobula_op.math.linalg_gemm, t, a = a, b = b, tA = False, tB = True)

    a = a.reshape((J, I))
    t = np.dot(a.T, b.T)
    check_math_func(mobula_op.math.linalg_gemm, t, a = a, b = b, tA = True, tB = True)

    b = b.T
    t = np.dot(a.T, b)
    check_math_func(mobula_op.math.linalg_gemm, t, a = a, b = b, tA = True, tB = False)
