class CustomOp(object):
    def __init__(self, *args, **kwargs):
        self._num_inputs = None
        self._num_outputs = None
        self.op = None
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def backward(self, *args, **kwargs):
        pass
    def infer_shape(self, in_shape):
        raise NotImplementedError

    @property
    def num_inputs(self):
        # [todo] variable inputs
        varnames = list(self.forward.__code__.co_varnames[1:])
        return len(varnames) if self._num_inputs is None else self._num_inputs

    @property
    def num_outputs(self):
        # [todo] variable outputs
        varnames = list(self.backward.__code__.co_varnames[1:])
        return len(varnames) if self._num_outputs is None else self._num_outputs

    '''
    def _get_varnames(self, func, num):
        varnames = list(func.__code__.co_varnames[1:])
        if num is None:
            return varnames
        if len(varnames) == 1:
            return ["%s_%d" % (vname, i) for i in range(num)]
        assert len(varnames) == num 
        return varnames 

    def list_arguments(self):
        return self._get_varnames(self.forward, self._num_inputs)

    def list_outputs(self):
        return self._get_varnames(self.backward, self._num_outputs)
    '''
