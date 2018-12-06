# MobulaOP

Linux | Windows | Coverage
------|---------|----------
[![Linux Build status](https://api.travis-ci.org/wkcn/MobulaOP.svg?branch=master)](https://travis-ci.org/wkcn/MobulaOP)|[![Windows Build status](https://ci.appveyor.com/api/projects/status/bvnavb8k2xnu0wqj?svg=true)](https://ci.appveyor.com/project/wkcn/mobulaop)|[![Coverage Status](https://coveralls.io/repos/github/wkcn/MobulaOP/badge.svg?branch=master)](https://coveralls.io/github/wkcn/MobulaOP?branch=master)

## What is it?
*MobulaOP* is a simple and flexible cross framework operators toolkit.

You can write custom operators by Python/C++/C/CUDA/HIP/TVM without rebuilding deep learning framework from source.

## How to use it?

[[中文教程](docs/tutorial-cn.md)]

[[Tutorial](docs/tutorial-en.md)]

- Add an addition operator [[Code](examples/MyFirstOP.py)]

```python
import mobula

@mobula.op.register
class MyFirstOP:
    def forward(self, x, y):
        return x + y
    def backward(self, dy): 
        return [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

# MXNet
import mxnet as mx
a = mx.nd.array([1,2,3])
b = mx.nd.array([4,5,6])
c = MyFirstOP(a, b)
print (c) # [5, 7, 9]

# NumPy
import numpy as np
a = np.array([1,2,3])
b = np.array([4,5,6])
op = MyFirstOP[np.ndarray]()
c = op(a, b)
print (c) # [5, 7, 9]

# PyTorch
import torch
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
c = MyFirstOP(a, b)
print (c) # [5, 7, 9]

```

- Use **custom operators** without rebuilding the source of deep learning framework [[Code](examples/RunROIAlign.py)]

```python
# Use ROIAlign operator
import mxnet as mx
import numpy as np
import mobula

# Load ROIAlign Module
mobula.op.load('ROIAlign')

ctx = mx.cpu(0)
dtype = np.float32
N, C, H, W = 2, 3, 4, 4

data = mx.nd.array(np.arange(N*C*H*W).astype(dtype).reshape((N,C,H,W)))
rois = mx.nd.array(np.array([[0, 1, 1, 3, 3]], dtype = dtype))

data.attach_grad()
with mx.autograd.record():
    # mx.nd.NDArray and mx.sym.Symbol are both available as the inputs.
    output = mobula.op.ROIAlign(data = data, rois = rois, pooled_size = (2,2), spatial_scale = 1.0, sampling_ratio = 1)

print (output.asnumpy(), data.grad.asnumpy())
```

- Import Custom C++ Operator Dynamically [[Code](examples/dynamic_import_op/dynamic_import_op.py)]

```python
import mobula
# Import Custom Operator Dynamically
mobula.op.load('./AdditionOP')

import mxnet as mx
a = mx.nd.array([1,2,3])
b = mx.nd.array([4,5,6])
c = mobula.op.AdditionOP(a, b)

print ('a + b = c \n {} + {} = {}'.format(a.asnumpy(), b.asnumpy(), c.asnumpy()))
```

## How to get it? 
```bash
# Clone the project
git clone https://github.com/wkcn/MobulaOP

# Enter the directory
cd MobulaOP

# Install Third-Party Library
pip install -r requirements.txt

# Build
sh build.sh

# Add MobulaOP into Enviroment Variable `PYTHONPATH`
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
