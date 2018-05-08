class CustomOp(object):
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    def backward(self, *args, **kwargs):
        pass
    def infer_shape(self, in_shape):
        raise NotImplementedError
