import os
import sys
import subprocess
import numpy as np

ENV_PATH = os.path.dirname(__file__)

if sys.version_info[0] < 3:
    FileNotFoundError = IOError
else:
    long = int


def asnumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, 'asnumpy'):
        return data.asnumpy()
    if hasattr(data, 'numpy'):
        return data.numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    raise TypeError('Unsupported Type: {}'.format(type(data)))


def assert_almost_equal(a, b, rtol=1e-5, atol=1e-8):
    def check_value(data, other):
        if isinstance(data, (int, long, float)):
            if hasattr(other, 'shape'):
                return np.full(other.shape, fill_value=data)
            else:
                return np.array(a)
        return data
    a = check_value(a, b)
    b = check_value(b, a)
    a = asnumpy(a)
    b = asnumpy(b)
    # Check Shape
    # If the shapes don't match, raise AssertionError and print the shapes
    assert a.shape == b.shape,\
        AssertionError('Unmatched Shape: {} vs {}'.format(a.shape, b.shape))

    # Compute Absolute Error |a - b|
    error = a - b
    abs_error = np.abs(error)
    max_abs_error = abs_error.max()

    def raise_error(abs_error, info):
        # tell where is maximum absolute error and the value
        loc = np.argmax(abs_error)
        idx = np.unravel_index(loc, abs_error.shape)
        out = ''

        def get_array_R(data, name, idx, R):
            axes = [-1] if data.ndim == 1 else [-1, -2]
            shape = data.shape
            slice_list = list(idx)
            sidx = list(idx[-2:])
            for i in axes:
                axis_len = shape[i]
                axis_i = slice_list[i]
                start = max(0, axis_i - R + 1)
                stop = min(axis_len, axis_i + R)
                slice_list[i] = slice(start, stop)
                sidx[i] -= start

            def str_slice_list(slice_list):
                return ', '.join([str(s) if not isinstance(s, slice) else
                                  '{}:{}'.format(s.start, s.stop) for s in slice_list])
            sdata = data.round(5)
            '''
            if data.ndim == 1:
                sdata[sidx[-1]] = str(sdata[sidx[-1]])
            else:
                sdata[sidx[-2]][sidx[-1]] = str(sdata[sidx[-2]][sidx[-1]])
            '''
            return '{name}[{slice_list}]:\n{data}\n'.format(name=name, slice_list=str_slice_list(slice_list),
                                                            data=sdata)

        R = 5
        out += 'Location of maximum error: {}\n'.format(idx)
        out += '{}\n{}\n{}'.format(info,
                                   get_array_R(
                                       a, 'a', idx, R),
                                   get_array_R(
                                       b, 'b', idx, R),
                                   )
        raise AssertionError(out)

    # Check Absolute Error
    if max_abs_error > atol:
        # If absolute error >= atol, raise AssertionError,
        idx = abs_error.argmax()
        raise_error(abs_error, 'Maximum Absolute Error({}) > atol({}): {} vs {}'.
                    format(max_abs_error, atol, a.ravel()[idx], b.ravel()[idx]))

    # Compute Relative Error |(a-b)/b|
    try:
        eps = np.finfo(b.dtype).eps
    except ValueError:
        eps = np.finfo(np.float32).eps
    relative_error = abs_error / (np.abs(b) + eps)
    max_relative_error = relative_error.max()

    # Check Relative Error
    if max_relative_error > rtol:
        # If relative error >= rtol, raise AssertionError,
        idx = relative_error.argmax()
        raise_error(relative_error, 'Maximum Relative Error({}) > rtol({}): {} vs {}'.
                    format(max_relative_error, rtol, a.ravel()[idx], b.ravel()[idx]))


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
        except:
            pass
    else:
        return range(0)
    return range(len([i for i in result.split('\n') if 'GPU' in i]))


def assert_file_exists(fname):
    assert os.path.exists(fname), IOError("{} not found".format(fname))


def get_git_hash():
    try:
        line = open(os.path.join(ENV_PATH, '..', '.git/HEAD')
                    ).readline().strip()
        ref = line[5:] if line[:4] == 'ref:' else line
        return open(os.path.join('.git', ref)).readline().strip()[:7]
    except FileNotFoundError:
        return 'custom'


FLT_MIN = 1.175494351e-38
FLT_MAX = 3.402823466e+38
