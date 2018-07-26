import os
import subprocess
import numpy as np

ENV_PATH = os.path.dirname(__file__)

def assert_almost_equal(a, b, atol = 1e-5, rtol = 1e-8):
    assert np.allclose(a, b, atol = atol, rtol = rtol), np.max(np.abs(a - b))

def list_gpus():
    """Return a list of GPUs
        Adapted from [MXNet](https://github.com/apache/incubator-mxnet)

    Returns
    -------
    list of int:
        If there are n GPUs, then return a list [0,1,...,n-1]. Otherwise returns
        [].
    """
    re = ''
    nvidia_smi = ['nvidia-smi', '/usr/bin/nvidia-smi', '/usr/local/nvidia/bin/nvidia-smi']
    for cmd in nvidia_smi:
        try:
            re = subprocess.check_output([cmd, "-L"], universal_newlines=True)
            break
        except:
            pass
    else:
        return range(0)
    return range(len([i for i in re.split('\n') if 'GPU' in i]))

def assert_file_exists(fname):
    assert os.path.exists(fname), IOError("{} not found".format(fname))

def get_git_hash():
    try:
        ref = open(os.path.join(ENV_PATH, '..', '.git/HEAD')).readline().split('ref:')[1].strip()
        return open(os.path.join('.git', ref)).readline().strip()[:7]
    except:
        return 'custom'

FLT_MIN = 1.175494351e-38
FLT_MAX = 3.402823466e+38
