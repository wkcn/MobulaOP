import mobula
import os
import numpy as np
import time
mobula.op.load('./EmptyOP', path=os.path.dirname(__file__))
EmptyOP = mobula.op.EmptyOP

T = np.float32
mobula.func.empty_forward.build('cpu', ['float'])

TIMES = 10000


def test_empty_op():
    a = np.array([1, 2, 3], dtype=T)
    tic = time.time()
    op = EmptyOP[np.ndarray]()
    for _ in range(TIMES):
        y = op(a)
    toc = time.time()

    used_time = toc - tic
    print('Used Time(s): %.3f' % used_time)


if __name__ == '__main__':
    test_empty_op()
