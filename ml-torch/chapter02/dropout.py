# 导入必要的库
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# hyper-parameters
num_samples = 20
hidden_size = 200
num_epochs = 500  # for train
torch.manual_seed(2333)

# generate data
x_train = torch.unsqueeze(torch.linspace(-1, 1, num_samples), 1)
y_train = x_train + 0.3 * torch.randn(num_samples, 1)
x_test = torch.unsqueeze(torch.linspace(-1, 1, num_samples), 1)
y_test = x_test + 0.3 * torch.randn(num_samples, 1)
# plot data
plt.scatter(x_train, y_train, c='r', alpha=0.5, label='train')
plt.scatter(x_test, y_test, c='b', alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2, 2))
plt.show()

# may be overfitting
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
)

# network with dropout
net_dropout = torch.nn.Sequential(
    torch.nn.Linear(1, hidden_size),
    torch.nn.Dropout(0.5),  # p=0.5
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.Dropout(0.5),  # p=0.5
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
)

# train
optimizer_overfitting = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_dropout = torch.optim.Adam(net_dropout.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(num_epochs):
    # forward propagation (predict, loss)
    pred_overfitting = net_overfitting(x_train)
    loss_overfitting = criterion(pred_overfitting, y_train)
    # backward propagation
    optimizer_overfitting.zero_grad()
    loss_overfitting.backward()
    optimizer_overfitting.step()

    # network with dropout
    pred_dropout = net_dropout(x_train)
    loss_dropout = criterion(pred_dropout, y_train)
    optimizer_dropout.zero_grad()
    loss_dropout.backward()
    optimizer_dropout.step()

# test
net_overfitting.eval()
net_dropout.eval()  # without dropout

test_pred_overfitting = net_overfitting(x_test)
test_pred_dropout = net_dropout(x_test)

plt.scatter(x_train, y_train, c='r', alpha=0.3, label='train')
plt.scatter(x_test, y_test, c='b', alpha=0.3, label='test')
plt.plot(x_test, test_pred_overfitting.data.numpy(), 'r-', lw=2, label='overfitting')
plt.plot(x_test, test_pred_dropout.data.numpy(), 'b--', lw=2, label='dropout')
plt.legend(loc='upper left')
plt.ylim((-2, 2))
plt.show()
