import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# hyper-parameters
num_epochs = 200

# Generate data (100 samples; y = x^2 + 1)
np.random.seed(32)
num_samples = 100
X = np.random.uniform(-5, 5, (num_samples, 1))  # 均匀分布
Y = X ** 2 + 1 + 5 * np.random.normal(0, 1, (num_samples, 1))  # 正态分布噪声
# to torch.float
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()
# plot scatter
plt.scatter(X, Y)
plt.show()

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=0)
train_dataloader = DataLoader(TensorDataset(train_X, train_Y), batch_size=32, shuffle=True)
test_dataloader = DataLoader(TensorDataset(test_X, test_Y), batch_size=32, shuffle=False)


class LinearRegression(nn.Module):
    """ underfitting """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """ normal """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)


class MLPOverfitting(nn.Module):
    """ overfitting """

    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        return self.output(x)


def plot_errors(models, num_epochs, train_dataloader, test_dataloader):
    loss_fn = nn.MSELoss()

    train_losses = []
    test_losses = []

    # Iterate through each type of model
    for model in models:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

        train_losses_per_model = []
        test_losses_per_model = []

        # train and test the model for num_epochs
        for epoch in range(num_epochs):

            # train
            model.train()
            train_loss = 0

            for inputs, targets in train_dataloader:
                # forward propagation (predict loss)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                # backward propagation
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # calculate average loss for the epoch
            train_loss /= len(train_dataloader)
            train_losses_per_model.append(train_loss)

            # test
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    # forward propagation (predict loss)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    test_loss += loss.item()

                # calculate average loss for the epoch
                test_loss /= len(test_dataloader)
                test_losses_per_model.append(test_loss)

        # return the train and test losses for each model
        train_losses.append(train_losses_per_model)
        test_losses.append(test_losses_per_model)

    return train_losses, test_losses


models = [LinearRegression(), MLP(), MLPOverfitting()]
train_losses, test_losses = plot_errors(models, num_epochs, train_dataloader, test_dataloader)

# visualize the train and test losses for each model
for i, model in enumerate(models):
    plt.figure(figsize=(8, 4))
    plt.plot(range(num_epochs), train_losses[i], label=f"Train {model.__class__.__name__}")
    plt.plot(range(num_epochs), test_losses[i], label=f"Test {model.__class__.__name__}")
    plt.legend()
    plt.ylim((0, 200))
    plt.savefig(f"output/demo{i}.png")
    plt.show()
