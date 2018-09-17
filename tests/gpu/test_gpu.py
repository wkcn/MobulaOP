import mxnet as mx
import sys
import os
from nose.core import TestProgram


def test_gpu():
    ctx = mx.gpu(0)
    mx.test_utils.set_default_context(ctx)
    assert mx.current_context() == ctx
    path = os.path.join(os.path.dirname(__file__), '../')
    TestProgram(defaultTest=path, argv=[path, '-s'])
    assert mx.current_context() == ctx


test_gpu()
