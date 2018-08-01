import sys
sys.path.append('../') # Add MobulaOP path
import mobula_op

@mobula_op.operator.register
class MyFirstOP:
    def forward(self, x, y):
        return x + y
    def backward(self, dy): 
        return [dy, dy]
    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]

import torch
a = torch.tensor([1,2,3], requires_grad = True)
b = torch.tensor([4,5,6], requires_grad = True)
c = MyFirstOP(a, b)
# c = a + b
print ('a + b = c \n {} + {} = {}'.format(a, b, c)) # [5, 7, 9]

d = c.sum()
d.backward()
print ('dc/da = {}\ndc/db = {}'.format(a.grad, b.grad)) # [1, 1, 1], [1, 1, 1]
