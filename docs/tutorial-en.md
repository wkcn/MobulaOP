# MobulaOP Tutorial

[MobulaOP](https://github.com/wkcn/MobulaOP) is a simple and flexible cross framework operators toolkit. You can create custom C++/C operators without rebuilding the source of deep learning framework. In addition, a code is enough to implement custom operators on CPU and GPU.

## Features
* 1. Simplification

The code of MobulaOP is tiny and easy to compile. You don't need rebuild the deep learning framwork to create custom C++/C operators any more.

* 2. Write once, run anywhere

Writting a code is enough to launch custom C++/C operators on different devices such as CPU/GPU, different deep learning frameworks like MXNet, PyTorch, and numerical computation libraries like NumPy.

* 3. Focus on high-level calculation for users

MobulaOP encapsulates the detail of low-level calculation. For users, they don't need to care about the detail, and they will focus on the core of algorithm.

* 4. Better for MXNet Custom Operator

There are more support for MXNet. You can create custom operators conveniently with MobulaOP.

# Installation
Open a terminal and input the following commands:
```bash
# Clone MobulaOP
git clone https://github.com/wkcn/MobulaOP
# Enter the project directory
cd MobulaOP
# Install the dependences: numpy, pyyaml and easydict
pip install -r requirements.txt
# Compile MobulaOP
sh build.sh
# Add the project directory into the environment variable `PYTHONPATH`
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

If installation success, there will be no prompt when inputting the command `python -c "import mobula"` outside the project directory.

# Kernel Function

After installation, it's available to write custom C++/C operators.

We call calculation function in parallel **Kernel Function**.

Taking element-wise product as an example:

```c++
template <typename T>
MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* out) {
    parfor(n, [&](int i) {
        out[i] = a[i] * b[i];
    });
}
```

The six lines of code implement a element-wise product calculation on CPU and GPU.

To be specific, the macro `MOBULA_KERNEL` declares this function as a kernel function. Kernel Function doesn't need returned value, and the suffix name of the function is `_kernel`.

For parameters list of kernel function, the first parameter must be the number of threads in parallel. The constant pointer `const T*` declares input tensors, and the variant pointer `T*` means the output tensors. The declaration of input/output tensors will affect the computation performance.

In the kernel function, `parfor` is a parallel for-loop. The first parameter of `parfor` is the number of iteration, and the second parameter is a function whose input is the subscript index of for-loop. We use a lambda function here. The subscript index `i` starts at 0, and `0 <= i < n`. MobulaOP will unroll the parfor-loop in different approaches for different devices. For example, MobulaOP will unroll the code in CPU device:

```c++
for (int i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
}
```

MobulaOP will use multi-threads, OpenMP, CUDA, etc to execute the loop **in parallel**.

Notice:

1. In `MOBULA_KERNEL` kernel function, the first element in the parameters list should be the number of threads in parallel.

2. The body of the parfor-loop will execute in parallel, so it's worth to notice **thread-safe** problem. MobulaOP provides `atomic_add` function for the atomic addition of CPU/GPU `float32` type.

3. In a kernel function, it's valid to call `parfor` multiple times. It allows to use different number of iteration of `parfor`s, but the numbers of threads are the same.

4. `parfor` should be called in kernel functions.

5. If you want to call other function in kernel functions, the called function should be declared as `MOBULA_DEVICE`, and own the returned type.

Example: Return the maximal value of two numbers
```c++
template <typename T>
MOBULA_DEVICE T maximum(const T a, const T b) {
    return a >= b ? a : b;
}
```

## Calling Kernel Function
We will call the aforementioned kernel function :)

MobulaOP will analyze kernel function, generate the corresponding code, and compile the code into dynamic link libraries.

Let us save the aforementioned kernel function into the file `MulElemWise.cpp`, and put the file into the following directory:
```bash
tutorial
└── MulElemWise
    └─── MulElemWise.cpp
```

Then create a file called `test_mul_func.py` in the directory `tutorial`, and write codes in this file.
```python
import mobula
mobula.op.load('MulElemWise')

import mxnet as mx
a = mx.nd.array([1,2,3])
b = mx.nd.array([4,5,6])
out = mx.nd.empty(a.shape)
mobula.func.mul_elemwise(a.size, a, b, out)
print (out)  # [4, 10, 18]
```

Enter the command `python test_mul_func.py` in the terminal.

The output is `[4, 10, 18]`.

The lines 1, 2, 8 of the code are important.

Line 1: Importing MobulaOP library.

Line 2: Loading the module `MulElemWise`. MobulaOP will search files `<module name>.cpp`, `<module name>.py` and `__init__.py`. if one of these files exists, these files will be compiled or loaded. It's available to pass an absolute path for `mobula.op.load`, e.g. `mobula.op.load('MulElemWise', os.path.dirname(__file__))`.

Line 8: Calling the kernel function `mul_elemwise`. It's noticed that there is no the postfix `_kernel` comparing with the function's declaration `MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* out)`.

Kernel functions will be added into `mobula.func`, and calling `mobula.func.<the name of kernel function>` means calling the C++/C kernel function.

MobulaOP can pre/post-processes the parameters automatically, such as getting the pointer of data, template instantiation, memory non-continuous tensor, calling `wait_to_read` or `wait_to_write` for tensors.

## Creating a custom operator
After writting a kernel function, we can encapsulate it into a custom operator. MobulaOP provides a simple method to declare it.

In the directory `tutorial/MulElemWise`, create the file `MulElemWise.py` and write the following code in it:

```python
import mobula

@mobula.op.register
class MulElemWise:
    def forward(self, a, b):
        mobula.func.mul_elemwise(a.size, a, b, self.y)
    def backward(self, dy):
        self.dX[0][:] = self.F.multiply(dy, self.X[1])
        mobula.func.mul_elemwise(dy.size, dy, self.X[0], self.dX[1])
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]
```

In line 3, `@mobula.op.register` is a Python decorator, registering the class as an operator.

In an operator, it's necessary to declare `forward`, `backward`, and `infer_shape` functions.

For the parameters list of the `forward` function, `a` and `b` are the inputs when feed-forwarding. For the parameters list of the `backward` function, `dy` is the input gradient when feed-backwarding.

MobulaOP will infer the number of inputs from `forward` function, and infer the number of outputs from `backward` function.

The `infer_shape` function accepts a tuple list, whose element is the shape of each input. There are two returned values, with the first value being the shape list of inputs, the second value being the shape list of outputs. the `infer_shape` function is similar with that in MXNet Python custom operator.

There are some built-in variables in the two functions `forward` and `backward`.

Variable Name    | Description
-----------------|--------------------------------------------
self.F           | context. e.g. self.F = mx.nd if using MXNet
self.X[k]        | the k-th input
self.Y[k]        | the k-th output
self.dX[k]       | the k-th input gradient
self.dY[k]       | the k-th output gradient
self.x           | the first input
self.y           | the first output
self.dx          | the first input gradient
self.dy          | the first output gradient
self.req[k]      | the assignment mode of the k-th tensor (null/write/add/replace)

It's noticed that `[:]` should be added when assigning a tensor, like `self.X[0][:] = data`

It's available to use the built-in function `assign` to assign a tensor, like `self.assign(self.X[0], self.req[0], data)`, which is consistent with with MXNet.

## Testing custom operators

Let's test the custom operator `MulElemWise` we write!

In the directory `tutorial`, create the file `test_mul_op.py` and write the following code:

```python
import mobula
mobula.op.load('MulElemWise')

import mxnet as mx
a = mx.nd.array([1,2,3])
b = mx.nd.array([4,5,6])

a.attach_grad()
b.attach_grad()
with mx.autograd.record():
    c = mobula.op.MulElemWise(a, b)
    c.backward()
    print (c)  # [4, 10, 18]
    print ('a.grad = {}'.format(a.grad.asnumpy()))  # [4, 5, 6]
    print ('b.grad = {}'.format(b.grad.asnumpy()))  # [1, 2, 3]
```

Enter `python test_mul_op.py` in the terminal, then we will get the result.

At line 2, MobulaOP loads the module `MulElemWise`, analyzes and registers the function into `mobula.func`. However the code is not compiled immediately.

At line 11 `c = mobula.op.MulElemWise(a, b)`, MobulaOP will determine the template instance given the types of inputs, then compile the corresponding dynamic link libraries, finally run the custom operator and return the result.

`mobula.op.MulElemWise` also accepts MXNet Symbol, NumPy NDArray, and PyTorch Tensor.

Examples:

MXNet Symbol:
```python
a_sym = mx.sym.Variable('a')
b_sym = mx.sym.Variable('b')
c_sym = mobula.op.MulElemWise(a_sym, b_sym)
```
NumPy NDArray:
```python
a_np = np.array([1,2,3])
b_np = np.array([4,5,6])
# NumPy doesn't record gradients, so create an operator instance to record it
op = mobula.op.MulElemWise[np.ndarray]()
c_np = op(a_np, b_np)
```

PyTorch Tensor:
```
a = torch.tensor([1, 2, 3], requires_grad=True)
b = torch.tensor([4, 5, 6], requires_grad=True)
c = mobula.op.MulElemWise(a, b)  # c = a + b
```

How to use these custom operators in Gluon?
```python
class MulElemWiseBlock(mx.gluon.nn.HybridBlock):
    def hybrid_forward(self, F, a, b):
        return mobula.op.MulElemWise(a, b)
```

The aforementioned codes can be seen at the docs directory of MobulaOP. [Code](https://github.com/wkcn/MobulaOP/tree/master/docs).

I hope that MobulaOP will help you :)

Any issue and pull request is welcome.

Thank you!
