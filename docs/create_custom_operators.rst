Create Custom Operators
=======================
It's flexible to create an operator in MobulaOP.

You only need to write a code, and run it on multiple frameworks.

- Notice

If mobula_op is not installed, you can add the MobulaOP path into sys.path.

For example:

.. code-block:: python

    import sys
    sys.path.append('./MobulaOP')

1. Create an operator with MobulaOP
-----------------------------------
Let's create an simple addition operator. :-)

.. code-block:: python

    import mobula_op

    # use the decorator to register an operator
    @mobula_op.operator.register
    class MyFirstOP:
        def forward(self, x, y):
            # return the forward result
            return x + y
        def backward(self, dy):
            # return the backward result
            return [dy, dy]
        def infer_shape(self, in_shape):
            # make sure that the shapes of the inputs are the same.
            assert in_shape[0] == in_shape[1]
            # return the shapes of inputs and outputs
            return in_shape, [in_shape[0]]

It's so easy!

For more details, please see it. [on going]

2. Run an operator with MobulaOP
--------------------------------
The operator with MobulaOP supports multiple framework.
In this note, we will run the addition operator in MXNet and NumPy.

- Run the addition operator in MXNet

.. code-block:: python

    import mxnet as mx
    a = mx.nd.array([1,2,3])
    b = mx.nd.array([4,5,6])
    c = MyFirstOP(a, b)
    print (c) # [5,7,9]

It's also available to pass a **mxnet.symbol.Symbol** into the operator.

- Run the addition operator in NumPy

.. code-block:: python

    import numpy as np
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    c = MyFirstOP(a, b)
    print (c) # [5,7,9]

For more details, please see it. [on going]
