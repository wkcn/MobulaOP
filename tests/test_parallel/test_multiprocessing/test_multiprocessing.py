import multiprocessing
import os
import time

import mobula
from mobula.testing import assert_almost_equal

mobula.op.load('./AddOp', os.path.dirname(__file__))


def call_op(i, q):
    import mxnet as mx
    n = 32
    x = mx.random.randint(-100, 100, (n,))
    y = mx.nd.zeros_like(x)
    mobula.func.add(n, x, i, y)
    assert_almost_equal(y, x + i)
    q.put(i)


def create_processes(num_processes):
    ps = []
    q = multiprocessing.Queue()
    for i in range(num_processes):
        p = multiprocessing.Process(target=call_op, args=(i, q))
        ps.append(p)
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    qs = []
    for i in range(num_processes):
        qs.append(q.get())
    qs.sort()
    for i in range(num_processes):
        assert qs[i] == i, qs


def test_multiprocessing():
    N = 16
    create_processes(N)
    time.sleep(1)
    create_processes(N)
