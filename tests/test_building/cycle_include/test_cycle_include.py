import os

import mobula

mobula.op.load('./Cycle', os.path.dirname(__file__))


def test_cycle_include():
    assert mobula.func.get_answer() == 42
