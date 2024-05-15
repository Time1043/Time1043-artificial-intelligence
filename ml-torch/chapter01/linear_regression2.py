import numpy as np
import torch
import torch.nn as nn

# hyper-parameters
input_dim = 1
output_dim = 1
learning_rate = 0.1  # for optimizer
num_epochs = 1000  # for training

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# generate some data (w=2, b=1)
np.random.seed(42)
x = np.random.rand(100, 1)
y = 1 + 2 * x + 0.1 * np.random.randn(100, 1)
# to pytorch tensor
x_tensor = torch.from_numpy(x).float().to(device)
y_tensor = torch.from_numpy(y).float().to(device)

# initialize weights and bias (model parameters)
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# model
model = nn.Linear(input_dim, output_dim).to(device)

# train
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # forward propagation
    y_pred = model(x_tensor)  # data to model
    loss = criterion(y_pred, y_tensor)

    # backward propagation
    optimizer.zero_grad()  # clear gradients
    loss.backward()
    optimizer.step()  # update weights and bias (model parameters)

print('w:', model.weight.data, '\nb:', model.bias.data)
