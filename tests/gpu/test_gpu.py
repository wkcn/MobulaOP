import mxnet as mx
import sys
import os
from nose.core import TestProgram 

def test_gpu():
    ctx = mx.gpu(0)
    mx.Context._default_ctx.value = ctx 
    path = os.path.join(os.path.dirname(__file__), '../')
    TestProgram(defaultTest = path)

test_gpu()
