import matplotlib.pyplot as plt
import numpy as np
import torch

# hyper-parameters
learning_rate = 0.1
num_epochs = 1000

# generate some data (w=2, b=1)
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)
# to pytorch tensor
x_tensor = torch.from_numpy(x).float()
y_tensor = torch.from_numpy(y).float()

# initialize weights and bias (model parameters)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# train
for epoch in range(num_epochs):
    # forward propagation
    y_pred = x_tensor * w + b
    loss = ((y_pred - y_tensor) ** 2).mean()

    # backward propagation
    loss.backward()

    # update weights and bias (model parameters)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # clear gradients for next iteration
        w.grad.zero_()
        b.grad.zero_()

# print final results
print('w:', w, '\nb:', b)

# visualize the results
plt.plot(x, y, 'o')
plt.plot(x_tensor.numpy(), y_pred.detach().numpy())
plt.show()
