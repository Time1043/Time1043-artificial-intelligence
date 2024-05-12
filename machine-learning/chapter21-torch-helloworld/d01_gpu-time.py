import torch
import time

print(torch.__version__)  # 2.0.1+cu118
print(torch.cuda.is_available())  # Ture

# Matrix multiplication in CPU mode
a = torch.randn(10000, 1000)
b = torch.randn(1000, 2000)

t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
print(a.device, t1 - t0, c.norm(2))

# Matrix multiplication in GPU mode
device = torch.device("cuda")
a = a.to(device)
b = b.to(device)

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))  # It takes time to activate the GPU device.

t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))  # Real running time

"""
cpu 0.22655916213989258 tensor(141234.9531)
cuda:0 7.067934513092041 tensor(141378.8125, device='cuda:0')
cuda:0 0.0010001659393310547 tensor(141378.8125, device='cuda:0')
"""
