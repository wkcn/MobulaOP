from mobula.test_utils import assert_almost_equal
import numpy as np


def check_almost_euqal_expection_raise(a, b, info, rtol=1e-5, atol=1e-8):
    try:
        assert_almost_equal(a, b, rtol=rtol, atol=atol)
        raise Exception(info)
    except AssertionError:
        pass


def test_almost_equal_shape():
    shape1 = (2, 2, 3)
    a = np.random.random(shape1)
    b = a.copy()
    c = a[1]
    assert_almost_equal(a, b)
    check_almost_euqal_expection_raise(a, c, "No exception raised")


def test_almost_equal_value():
    shape1 = (2, 2, 3)
    a = np.random.random(shape1)
    b = a.copy()
    atol = 1e-3
    assert_almost_equal(a, b, atol=0)
    assert_almost_equal(a, b, atol=atol)
    b[0, 0, 0] += atol
    b[0, 1, 2] -= atol
    assert_almost_equal(a, b, rtol=np.inf, atol=atol * 2.0)
    check_almost_euqal_expection_raise(
        a, b, 'Absolute Error Check failed', rtol=np.inf, atol=atol/2.0)
    eps = np.finfo(b.dtype).eps
    rtol = np.max(np.abs((a - b) / (b + eps)))
    assert_almost_equal(a, b, rtol=rtol * 2.0, atol=atol * 2.0)
    check_almost_euqal_expection_raise(
        a, b, 'Relative Error Check failed', rtol=rtol * 2.0, atol=atol / 2.0)
