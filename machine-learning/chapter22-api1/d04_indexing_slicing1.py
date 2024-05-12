import torch

a = torch.randn(4, 3, 28, 28)  # 4 batch x 3 channel x  28 height x 28 width
print(a[0].shape)  # torch.Size([3, 28, 28])  # a colour picture
print(a[0, 0].shape)  # torch.Size([28, 28])  # a grayscale picture
print(a[0, 0, 2, 4])  # tensor(0.5465)  # a single pixel value

# slice: start:end:step
print(a[:2].shape)  # torch.Size([2, 3, 28, 28])  # first 2 pictures
print(a[:2, -1, :, :].shape)  # torch.Size([2, 28, 28])  # first 2 pictures, last channel
print(a[:, :, 0:28:2, ::2].shape)  # torch.Size([4, 3, 14, 14])  # pixel select by step

# select by specific index
print(a.index_select(0, torch.tensor([0, 2])).shape)  # torch.Size([2, 3, 28, 28])  # select pictures 0 and 2
print(a.index_select(1, torch.tensor([0, 2])).shape)  # torch.Size([4, 2, 28, 28])  # select channel 0 and 2
print(a.index_select(2, torch.arange(8)).shape)  # torch.Size([4, 3, 8, 1])  # select 8 pixels in height dimension

# ... is a shortcut for all dimensions
print(a[...].shape)  # torch.Size([4, 3, 28, 28])  # all pictures
print(a[0, ...].shape)  # torch.Size([3, 28, 28])  # first picture
print(a[:, 1, ...].shape)  # torch.Size([4, 28, 28])  # second channel of all pictures

# select by mask
x = torch.randn(3, 4)
mask = x.ge(0.5)  # mask for values greater than 0.5
print(x, "\n", mask)
print(torch.masked_select(x, mask))
"""
tensor([[ 0.5663,  0.3780, -1.1873,  0.3202],
        [-0.4486,  0.3809, -0.8121,  0.5216],
        [ 0.3068,  1.7174,  0.4122,  1.0049]]) 
tensor([[ True, False, False, False],
        [False, False, False,  True],
        [False,  True, False,  True]])
tensor([0.5663, 0.5216, 1.7174, 1.0049])
"""

# select by flattened index
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(torch.take(src, torch.tensor([0, 2, 5])))  # tensor([4, 5, 8])
