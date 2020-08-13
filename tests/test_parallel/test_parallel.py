import ctypes
import os

import numpy as np

import mobula
from mobula.testing import assert_almost_equal, gradcheck

mobula.op.load('./Parallel', os.path.dirname(__file__))


def test_sync():
    N = 10000
    x = np.zeros((1, ), dtype=np.int32)
    mobula.func.test_syncthreads(N, x)
    assert x[0] == N, (x[0], N)


def test_parfor():
    N = 10000
    x = np.empty((N, ), dtype=np.int32)
    mobula.func.test_parfor(N, x)
    assert (x == np.arange(N).astype(np.int32)).all()


if __name__ == '__main__':
    test_sync()
    test_parfor()
