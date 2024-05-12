import numpy as np
import torch

# Dimension 0
# applied to the result of loss function, return a scalar!!
a = torch.tensor(1.)
print(a.dim(), len(a.shape))  # 0
print(a.size())  # torch.Size([])
print(a.shape)  # torch.Size([])

# Dimension 1
# applied to bias, return a vector!!
# applied to Linear Input, return a vector!!
b = torch.tensor([1.1])  # accept a list
c = torch.FloatTensor(2)  # accept the shape of data (2,)
d = torch.from_numpy(np.array([1.2, 2.3]))  # convert from numpy array
print(b)  # tensor([1.1000])
print(c)  # tensor([0., 0.])
print(d)  # tensor([1.2000, 2.3000], dtype=torch.float64)

# Dimension 2
# applied to Linear Input Batch, return a matrix!!
e = torch.randn(2, 3)  # create a random tensor with shape (2, 3)
print(e.dim(), len(e.shape))  # 2
print(e.size(), e.size(0), e.size(1))  # torch.Size([2, 3])  2  3
print(e.shape, e.shape[0], e.shape[1])  # torch.Size([2, 3])  2  3

# Dimension 3
# applied to RNN Input Batch, return a tensor with 3 dimensions!!
f = torch.rand(2, 2, 3)  # create a random tensor with shape (2, 2, 3)
print(f.numel())  # number of elements (2*2*3)

# Dimension 4
# applied to CNN Input Batch, return a tensor with 4 dimensions!!
g = torch.rand(2, 3, 28, 28)  # create a random tensor with shape (2, 3, 28, 28)
print(g.numel())  # number of elements (2*3*28*28)
