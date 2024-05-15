import torch
import matplotlib.pyplot as plt

# hyperparameters
learning_rate = 0.01
num_epochs = 100

# generate data
X = torch.randn(100, 1)
y = 2 * X + 3 + torch.randn(100, 1)

# initialize parameters
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer
optimizer = torch.optim.Adagrad([w, b], lr=learning_rate)
losses = []  # record loss

# 训练模型
for epoch in range(num_epochs):
    # forward prediction
    y_pred = w * X + b  # predicted value
    loss = torch.mean((y_pred - y) ** 2)  # mean squared error loss
    losses.append(loss.item())
    optimizer.zero_grad()

    # backward propagation
    loss.backward()
    optimizer.step()

# plot loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
