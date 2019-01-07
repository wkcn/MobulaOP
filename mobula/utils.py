import os
import subprocess

ENV_PATH = os.path.dirname(__file__)


def list_gpus():
    """Return a list of GPUs
        Adapted from [MXNet](https://github.com/apache/incubator-mxnet)

    Returns
    -------
    list of int:
        If there are n GPUs, then return a list [0,1,...,n-1]. Otherwise returns
        [].
    """
    result = ''
    nvidia_smi = ['nvidia-smi', '/usr/bin/nvidia-smi',
                  '/usr/local/nvidia/bin/nvidia-smi']
    for cmd in nvidia_smi:
        try:
            result = subprocess.check_output(
                [cmd, "-L"], universal_newlines=True)
            break
        except Exception:
            pass
    else:
        return range(0)
    return range(len([i for i in result.split('\n') if 'GPU' in i]))


def get_git_hash():
    try:
        GIT_HEAD_PATH = os.path.join(ENV_PATH, '..', '.git')
        line = open(os.path.join(GIT_HEAD_PATH, 'HEAD')
                    ).readline().strip()
        if line[:4] == 'ref:':
            ref = line[5:]
            return open(os.path.join(GIT_HEAD_PATH, ref)).readline().strip()[:7]
        return line[:7]
    except FileNotFoundError:
        return 'custom'
