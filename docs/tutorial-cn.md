# 这可能是创建自定义C++ Operator最简单的方式 - MobulaOP使用说明

大家好，我想在这里给大家介绍我的一个项目：[MobulaOP](https://github.com/wkcn/MobulaOP).

MobulaOP是一个简单且灵活的跨框架算子创建工具包。不需要重新编译深度学习框架的源码，就可以创建自定义的C++算子。而且只需要一份C++代码实现和简单的定义，自定义算子就可以在CPU和GPU上运行。

之所以建立这个项目，是因为我发现MXNet创建自定义算子的方法不太方便，其他深度学习框架也同样存在这个问题。

当前，创建自定义算子的方法主要有：
- 1. 重新编译深度学习框架的源码
重新编译源码耗时过长。需要了解对应框架的算子实现形式，编写出的代码不适用于其他框架。
- 2. 使用运行时编译(Run-Time Compilation)API 
需要编写对应的CUDA代码，编写过程较复杂，无法在CPU环境下进行调试。
- 3. 加载动态文件
需要了解对应框架的动态加载实现形式，编写较复杂，一份代码不适用于多个框架。

因此，我设计了MobulaOP项目，希望能解决上述问题。

MobulaOP项目当前的特性有：
- 1. 项目实现精简，不需要重新编译深度学习框架，就可以实现自定义的C++ operator;
- 2. 只需要编写一份代码，就可以让自定义算子运行在不同设备(CPU/GPU)，以及不同的深度学习框架(如MXNet, PyTorch)或数值计算库NumPy上；
- 3. 在编写自定义层的过程中，用户有更多的注意力关注在运算的实现上；
- 4. 对MXNet有更多的支持，使用MobulaOP可以更方便地创建自定义算子(Custom Operator).

MobulaOP支持Linux、Windows和MacOS系统。

下面，我想简单地介绍一下[MobulaOP](https://github.com/wkcn/MobulaOP)的使用方法。

## 配置MobulaOP
在终端下输入以下命令:
```bash
# 将MobulaOP项目拷贝下来
git clone https://github.com/wkcn/MobulaOP
# 进入项目文件夹
cd MobulaOP
# 安装依赖库numpy, pyyaml和easydict
pip install -r requirements.txt
# 进行编译
sh build.sh
# 将MobulaOP文件夹加入PYTHONPATH环境变量中
export PYTHONPATH=$PYTHONPATH:$(pwd)
```
当执行完以上命令后，在项目目录外打开Python交互界面，输入`import mobula`，如果没有提示，则表示配置成功。

## 核函数

配置好MobulaOP后，就可以使用C++编写算子(operator)的运算函数了。

这里把并行计算的运算函数称为**核函数**。

以创建一个逐位乘法算子为例，它的实现为：

```c++
template <typename T>
MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* out) {
    parfor(n, [&](int i) {
        out[i] = a[i] * b[i];
    });
}
```

没错，定义一个逐位乘法函数只需要6行代码，并且它支持在CPU和GPU下运行。

其中，`MOBULA_KERNEL`宏声明了这个函数是一个核函数。核函数不需要定义返回值，同时核函数的函数名后缀为`_kernel`.

对于参数列表，MobulaOP要求第一个参数为并行计算的线程数。MobulaOP会自动将参数列表中`const T*`类型的参数识别为输入数组的指针，将`T*`类型的参数识别为输出数组的指针。

函数块中，调用了并行执行的`parfor`循环函数。这个函数的第一个参数为循环体的总迭代数，第二个参数为一个接收迭代下标的函数，这里使用了匿名函数。下标`i`从0开始计数，满足`0 <= i < n`。MobulaOP会根据运行设备对`parfor`进行不同的展开。当这段代码在CPU下运行时，MobulaOP会将这段函数展开为：

```c++
for (int i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
}
```
MobulaOP会自动地使用多线程、OpenMP、CUDA等方法**并行**地执行这个循环。

需要注意的是：

1. `MOBULA_KERNEL`核函数的第一个参数为调用这个函数进行并行计算的线程数；
2. 核函数内部语句均为并行执行，编写核函数时要**注意线程安全问题**。当前，MobulaOP提供了CPU/GPU下单精度浮点数(float32)的`atomic_add`原子加函数；
3. 在一个核函数内，允许多次调用`parfor`函数, 这些`parfor`的总迭代数可以不同，但实际使用的线程数是相同的；
4. `parfor`函数只允许在核函数内部进行调用；
5. 如果要在核函数中调用其他函数，被调用的函数的声明前需要添加宏`MOBULA_DEVICE`, 并声明返回值类型。

例子：返回两个数中的最大值
```c++
template <typename T>
MOBULA_DEVICE T maximum(const T a, const T b) {
    return a >= b ? a : b;
}
```

## 执行核函数
接下来，使用MobulaOP执行上述核函数。

MobulaOP能够自动分析、生成代码，并调用编译器将代码编译为动态链接库。

把上述核函数保存为`MulElemWise.cpp`文件，放在如下的文件目录结构中:
```bash
tutorial
└── MulElemWise
    └─── MulElemWise.cpp
```
在`tutorial`文件夹下创建`test_mul_func.py`文件，在这个文件中编写Python代码：
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

在终端中输入`python test_mul_func.py`即可执行。

这段代码中，与MobulaOP相关的一共有三行(第1、2、8行) 

第1行代码导入MobulaOP包。

第2行代码加载`MulElemWise`模块。MobulaOP会搜索`MulElemWise`文件夹中是否存在同名的`.cpp`或`.py`文件，以及`__init__.py`文件。若找到这些文件，将会对文件进行编译或加载。`mobula.op.load`也支持指定搜索目录，如`mobula.op.load('MulElemWise', os.path.dirname(__file__))`.

第8行调用核函数`mul_elemwise`，与函数声明`MOBULA_KERNEL mul_elemwise_kernel(const int n, const T* a, const T* b, T* out)`相比，在Python中调用的函数名比C++中的函数名少了后缀`_kernel`. MobulaOP把加载后的核函数添加到`mobula.func`中，调用`mobula.func.<核函数名>`即可调用C++函数。MobulaOP能够自动对参数进行处理, 包括获取数据指针、选择参数模板、处理内存非连续数组、根据参数的输入输出类型自动调用`wait_to_read`、`wait_to_write`函数等。

## 创建自定义算子(operator)
如何将核函数封装成一个算子(operator)呢，MobulaOP提供了一个简单的声明方法。
在`tutorial/MulElemWise`文件夹下创建文件`MulElemWise.py`, 输入以下代码：
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

第3行的`@mobula.op.register`为一个Python装饰器，它将其下面的类注册为算子。

一个算子类需要定义`forward`, `backward`以及`infer_shape`函数。

在`forward`函数的参数列表中，`a`和`b`是算子前向传播的输入；在`backward`函数的参数列表中，`dy`为算子后向传播时输入的导数。

MobulaOP会根据`forward`函数得到算子的输入个数和名称，根据`backward`得到输出个数。

`infer_shape`函数传入的是元组(tuple)的列表，分别表示各输入的尺寸(shape). `infer_shape`的返回值有两个值，第一个值是各个输入的尺寸，第二个值是各个输出的尺寸。`infer_shape`和MXNet自定义层里的`infer_shape`是相似的。

在算子的`forward`和`backward`函数中，定义了一些变量：

变量名     | 描述
-----------|-----------------------------------------
self.F     | 当前环境。假如使用MXNet, self.F = mx.nd
self.X[k]  | 第k个输入
self.Y[k]  | 第k个输出
self.dX[k] | 第k个输入的导数
self.dY[k] | 第k个输出的导数
self.x     | 第1个输入
self.y     | 第1个输出
self.dx    | 第1个输入的导数
self.dy    | 第1个输出的导数
self.req[k]| 第k个输入/输出的处理模式(null/write/add/replace)

值得注意的是，当使用一个数组或数字对另一个数组赋值时，被赋值的变量后面需要加上`[:]`，如`self.X[0][:] = data`

我们也可以使用内置的`assign`函数进行赋值，如`self.assign(self.X[0], self.req[0], data)`, 这里的`assign`函数和MXNet是一致的。

## 测试自定义算子

编写好`MulElemWise`算子的定义后，来测试一下吧。

在`tutorial`文件夹下创建文件`test_mul_op.py`, 输入代码：

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

同样，在终端输入`python test_mul_op.py`指令执行。

这里与MobulaOP有关的新代码是第11行: `c = mobula.op.MulElemWise(a, b)`

MobulaOP加载`MulElemWise`模块后，分析了`MulElemWise`文件夹下的`MulElemWise.cpp`文件，把核函数注册到`mobula.func`中；同时加载同一个文件夹下的`MulElemWise.py`文件，将算子注册到`mobula.op`中。这个过程没有发生编译。

当`mobula.op.MulElemWise(a, b)`执行时，MobulaOP会根据变量类型，自动编译所需要的动态链接库，并返回结果。

`mobula.op.MulElemWise`也可以接受MXNet的符号(Symbol)、NumPy数组或PyTorch Tensor.

例子：
MXNet的符号(Symbol): 
```python
a_sym = mx.sym.Variable('a')
b_sym = mx.sym.Variable('b')
c_sym = mobula.op.MulElemWise(a_sym, b_sym)
```

NumPy数组：
```python
a_np = np.array([1,2,3])
b_np = np.array([4,5,6])
# 由于NumPy不支持记录梯度，因此需要一个实例记录梯度
op = mobula.op.MulElemWise[np.ndarray]()
c_np = op(a_np, b_np)
```

PyTorch Tensor:
```
a = torch.tensor([1, 2, 3], requires_grad=True)
b = torch.tensor([4, 5, 6], requires_grad=True)
c = mobula.op.MulElemWise(a, b)  # c = a + b
```

如何在Gluon内使用MobulaOP定义的算子呢？

我们可以这样写：
```python
class MulElemWiseBlock(mx.gluon.nn.HybridBlock):
    def hybrid_forward(self, F, a, b):
        return mobula.op.MulElemWise(a, b)
```

这就是MobulaOP的简单使用介绍，上述代码可以在项目的文档部分(docs)[查看](https://github.com/wkcn/MobulaOP/tree/master/docs)。

希望MobulaOP能够对大家有帮助。

同时，欢迎大家对MobulaOP项目提Issue和PR. 谢谢！
