import os
import sys
import numpy as np
from .glue.common import MobulaOperator

if sys.version_info[0] < 3:
    FileNotFoundError = IOError
else:
    long = int


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, 'asnumpy'):
        return data.asnumpy()
    if hasattr(data, 'numpy'):
        return data.numpy()
    if isinstance(data, (list, tuple)):
        return np.array(data)
    raise TypeError('Unsupported Type: {}'.format(type(data)))


def to_tuple(data):
    if isinstance(data, tuple):
        return data
    if isinstance(data, list):
        return tuple(data)
    return (data, )


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
    a = to_numpy(a)
    b = to_numpy(b)
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
    if atol is not None:
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


def assert_file_exists(fname):
    assert os.path.exists(fname), IOError("{} not found".format(fname))


def gradcheck(func, inputs, kwargs=None, eps=1e-6, rtol=1e-2, atol=None, sampling=None):
    assert isinstance(func, MobulaOperator)
    if kwargs is None:
        kwargs = dict()
    assert isinstance(kwargs, dict)
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs, )
    # To NumPy Tensor
    inputs = [to_numpy(x) for x in inputs]
    func = func[np.ndarray](**kwargs)
    ori_out = to_tuple(func(*inputs))
    assert isinstance(ori_out, (tuple, list)), type(ori_out)
    dys = [np.random.normal(0, 0.01, size=out_i.shape) +
           0.1 for out_i in ori_out]
    assert len(dys) == len(ori_out), '{} vs {}'.format(len(dys), len(ori_out))
    grad = to_tuple(func.backward(dys))
    for i, x in enumerate(inputs):
        size = inputs[i].size
        sample_grad = np.empty_like(inputs[i])
        sample_grad_ravel = sample_grad.ravel()
        samples = np.arange(size)
        if sampling is not None:
            if isinstance(sampling, int):
                num_samples = sampling
            else:
                num_samples = int(sampling * size)
            samples = np.random.choice(samples, min(size, num_samples))
        for k in samples:
            x_ravel = x.ravel()
            old_elem_value = x_ravel[k]
            x_ravel[k] = old_elem_value + eps / 2
            pos_out = to_tuple(func(*inputs, **kwargs))
            x_ravel[k] = old_elem_value - eps / 2
            neg_out = to_tuple(func(*inputs, **kwargs))
            assert len(pos_out) == len(neg_out)
            assert len(pos_out) == len(ori_out)
            numerical_grad_k = np.sum(
                [dy * (pos - neg) / eps for pos, neg, dy in zip(pos_out, neg_out, dys)])
            sample_grad_ravel[k] = numerical_grad_k
        numerical_grad = grad[i].copy()
        numerical_grad.ravel()[samples] = sample_grad.ravel()[samples]
        assert_almost_equal(numerical_grad, grad[i], rtol=rtol, atol=atol)
