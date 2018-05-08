# MobulaOP

[![](https://api.travis-ci.org/wkcn/MobulaOP.svg?branch=master)](https://travis-ci.org/wkcn/MobulaOP)
[![Coverage Status](https://coveralls.io/repos/github/wkcn/MobulaOP/badge.svg?branch=master)](https://coveralls.io/github/wkcn/MobulaOP?branch=master)

## What is it?
*MobulaOP* is a simple and flexible cross framework operators toolkit.

You can write the custom operators by Python/C++/C/CUDA without rebuilding deep learning framework from source.

## How to use it?

- Add an addition operator

```python
import mobula

@mobula.operator.register('MyFirstOP')
class MyFirstOP(mobula.operator.CustomOp):
    def forward(self, x, y):
        return x + y
    def backward(self, dy): 
        return [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

import mxnet as mx
a = mx.nd.array([1,2,3]) 
b = mx.nd.array([4,5,6])
c = MyFirstOP(a, b)
print (c) # [5,7,9]
```
