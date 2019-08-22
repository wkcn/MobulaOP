import mobula
from mobula.testing import assert_almost_equal, gradcheck
import numpy as np
from nose.tools import assert_raises


def check_almost_euqal_expection_raise(a, b, info, rtol=1e-5, atol=1e-8):
    assert_raises(AssertionError, assert_almost_equal,
                  a, b, rtol=rtol, atol=atol)


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
        a, b, 'Absolute Error Check failed', rtol=np.inf, atol=atol / 2.0)
    eps = np.finfo(b.dtype).eps
    rtol = np.max(np.abs((a - b) / (b + eps)))
    assert_almost_equal(a, b, rtol=rtol * 2.0, atol=atol * 2.0)
    check_almost_euqal_expection_raise(
        a, b, 'Relative Error Check failed', rtol=rtol * 2.0, atol=atol / 2.0)


def test_gradcheck():
    @mobula.op.register
    class SquareOP:
        def __init__(self, err=0):
            self.err = err

        def forward(self, x):
            return self.F.square(x)

        def backward(self, dy):
            return 2 * self.x * dy + self.err

        def infer_shape(self, in_shape):
            assert len(in_shape) == 1
            return in_shape, in_shape
    shape = (3, 4, 5, 6)
    gradcheck(SquareOP, [np.random.normal(size=shape)], dict(err=0))
    assert_raises(AssertionError, gradcheck, SquareOP, [
                  np.random.normal(size=shape)], dict(err=100))

    @mobula.op.register
    class SingleOutput:
        def forward(self, x):
            ar = self.F.arange(x.size)
            return (x * ar + ar).sum() / (x.size - 1)

        def backward(self, dy):
            ar = self.F.arange(self.x.size)
            return (ar * dy) / (self.x.size - 1)

        def infer_shape(self, in_shape):
            return in_shape, [(1, )]

    gradcheck(SingleOutput, [np.random.normal(size=(64, ))])

    @mobula.op.register
    class TwoOutput:
        def forward(self, x, y):
            return x * y, x + y

        def backward(self, da, db):
            return da * self.X[1] + db, da * self.X[0] + db

        def infer_shape(self, in_shape):
            return in_shape, in_shape

    gradcheck(TwoOutput, [np.random.normal(
        size=shape), np.random.normal(size=shape)])

    gradcheck(TwoOutput, [np.random.normal(
        size=shape), np.random.normal(size=shape)], sampling=0.8)

    gradcheck(TwoOutput, [np.random.normal(
        size=shape), np.random.normal(size=shape)], sampling=10)
