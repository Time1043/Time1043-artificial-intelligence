import numpy as np
import torch

# Create a tensor [0,1)
a = torch.rand(2, 3, 4)  # shape
b = torch.rand_like(a)
# Create a tensor [min, max)
c = torch.randint(10, 20, (2, 3, 4))

# normal distribution  N~(0,1)
d = torch.randn(2, 3, 4)
# normal distribution  N~(mean, std)
e = torch.normal(mean=10, std=1, size=(2, 3, 4))
f = torch.normal(mean=torch.full([10], 0.0), std=torch.arange(1, 0, -0.1).float())

g = torch.eye(3)  # identity matrix

h = torch.full((2, 3, 4), 10)  # full
i = torch.full([2, 3, 4], 10)
j = torch.full([], 10)  # scalar tensor(10)
k = torch.full([1], 10)  # vector tensor([10])

# arange / range (no recommended)
l = torch.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
m = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
n = torch.arange(10, 0, -1)  # [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

# linspace / logspace
o = torch.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]
p = torch.logspace(0, 1, 3)  # tensor([ 1.0000,  3.1623, 10.0000])

'''随机种子shuffle 
torch.randperm(n, *, generator=None, out=None, 
dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False, pin_memory=False) 
-> Tensor'''
a = torch.randperm(10)
'''应用如N-Batch'''
train_data = torch.randn(4, 5)
train_label = torch.rand(4, 1)
index = torch.randperm(4)

train_data_shuffle = train_data[index]
train_label_shuffle = train_label[index]
