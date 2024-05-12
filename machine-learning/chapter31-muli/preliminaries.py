import torch

# y = 2 * X^T * X
x = torch.arange(4.0)  # tensor([0., 1., 2., 3.])
x.requires_grad_(True)  # X = torch.arange(4.0, requires_grad=True)  save gradient
print(x.grad)  # None

# calculate
y = 2 * torch.dot(x, x)
print(y)  # tensor(28., grad_fn=<MulBackward0>) - autograd

# backward
y.backward()
print(x.grad)  # tensor([ 0.,  4.,  8., 12.])

# check
print(x.grad == 4 * x)  # tensor([True, True, True, True])

# default pytorch accumulates gradients, so we need to zero them manually
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)  # tensor([1., 1., 1., 1.])
