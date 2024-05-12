import numpy as np
import torch

# Generates uninitialized data
f = torch.empty(2, 3)  # receives shape, data is random (inf or nan!)
g = torch.FloatTensor(2, 3)  # receives shape, data is random (inf or nan!)

# Default dtype is float32
# Reinforcement learning always uses double, requesting more precision
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1, 2.3]).type())  # torch.DoubleTensor
