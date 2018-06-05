import numpy as np
def assert_almost_equal(a, b, atol = 1e-5, rtol = 1e-8):
    assert np.allclose(a, b, atol = atol, rtol = rtol), np.max(np.abs(a - b))

FLT_MIN = 1.175494351e-38
FLT_MAX = 3.402823466e+38
