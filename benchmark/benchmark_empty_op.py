import mobula
import os
import time
import mxnet as mx
import numpy as np
mobula.op.load('./EmptyOP', path=os.path.dirname(__file__))
EmptyOP = mobula.op.EmptyOP

T = np.float32
mobula.func.empty_forward.build('cpu', ['float'])

TIMES = 3000


def test_empty_op():
    a = np.array([1, 2, 3], dtype=T)
    tic = time.time()
    op = EmptyOP[np.ndarray]()
    for _ in range(TIMES):
        a = op(a)
    toc = time.time()

    used_time = toc - tic
    print('Used Time(s): %.3f' % used_time)


def test_hybrid_op():
    class TestBlock(mx.gluon.block.HybridBlock):
        def hybrid_forward(self, F, x):
            return EmptyOP(x)
    block = TestBlock()
    a = mx.nd.array([1, 2, 3], dtype=T)

    # prepare
    for _ in range(10):
        a = block(a)
    mx.nd.waitall()

    a = mx.nd.array([1, 2, 3], dtype=T)
    tic = time.time()
    for _ in range(TIMES):
        a = block(a)
    mx.nd.waitall()
    toc = time.time()
    used_time = toc - tic
    print('Imperative Time(s): %.3f' % used_time)

    block.hybridize()
    # prepare
    for _ in range(10):
        a = block(a)
    mx.nd.waitall()

    a = mx.nd.array([1, 2, 3], dtype=T)
    tic = time.time()
    for _ in range(TIMES):
        a = block(a)
    mx.nd.waitall()
    toc = time.time()
    used_time = toc - tic
    print('Symbolic Time(s): %.3f' % used_time)


if __name__ == '__main__':
    test_empty_op()
    test_hybrid_op()
