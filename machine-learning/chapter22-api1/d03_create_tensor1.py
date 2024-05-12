import numpy as np
import torch

# create a tensor with ndarray
# arr = np.ones((2, 3))
arr = np.array([1, 2.3])
a = torch.from_numpy(arr)
print(a)  # tensor([1.0000, 2.3000], dtype=torch.float64)

# create a tensor with list
b = torch.tensor([[1, 2.3], [4, 5]])
print(b)  # tensor([[1.0000, 2.3000], [4.0000, 5.0000]])

# Lowercase tensor receives the list of data!!!
c = torch.tensor([3, 2], dtype=torch.float32)  # tensor([3., 2.])
# uppercase Tensor receives shape!!!
d = torch.FloatTensor(3, 2)  # tensor([[0., 0.], [0., 0.], [0., 0.]])
# also receives the list of data (not recommend)
e = torch.FloatTensor([3, 2])  # tensor([3., 2.])
