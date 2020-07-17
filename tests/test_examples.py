import mobula
import importlib
import runpy
import os
import sys


def packages_exist(pkg_names):
    for pkg in pkg_names:
        try:
            importlib.import_module(pkg)
        except ImportError:
            return False
    return True


def test_examples():
    EXAMPLES_PATH = os.path.join(os.path.dirname(__file__), '../examples')
    examples = [
        ([], ['MyFirstOP.py', '../docs/tutorial/test_mul_func.py',
              '../docs/tutorial/test_mul_op.py']),
        (['mxnet'], ['RunROIAlign.py',
                     'dynamic_import_op/dynamic_import_op.py',
                     '../opzoo/Convolution/test_conv.py',
                     # '../opzoo/Softmax/test_softmax.py',
                     '../opzoo/ROIAlign/test_roialign.py',
                     '../opzoo/Sum/test_sum.py',
                     '../opzoo/Transpose/test_transpose.py',
                     ]),
        (['mxnet', 'tvm', 'topi'], ['TVMOp.py']),
    ]
    sys.path.append('./')
    record = []
    for dep_pkgs, examples in examples:
        if packages_exist(dep_pkgs):
            for example in examples:
                print('testing... {}'.format(example))
                subpath, script_name = os.path.split(example)
                fullpath = os.path.join(EXAMPLES_PATH, subpath)
                old_workpath = os.getcwd()
                os.chdir(fullpath)
                try:
                    runpy.run_path(script_name, {}, '__main__')
                except Exception as e:
                    record.append((example, e))
                os.chdir(old_workpath)
    assert len(record) == 0, record
