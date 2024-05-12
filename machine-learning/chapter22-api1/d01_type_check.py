import torch

# Type check
data = torch.randn(2, 3)
print(isinstance(data, torch.FloatTensor))  # True
print(data.type())  # torch.FloatTensor
print(type(data))  # <class 'torch.Tensor'>  no use!!

# GPU and CPU tensors are different types
print(isinstance(data, torch.cuda.FloatTensor))  # False
data = data.cuda()  # move to GPU, return a reference on the GPU
print(isinstance(data, torch.cuda.FloatTensor))  # True
