import os

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision  # cv
from matplotlib import pyplot as plt
from mnist_utils import plot_curve, plot_image, one_hot

# set the project path
project_path = os.path.dirname(__file__)
data_path = os.path.join(project_path, 'data')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # plot_image()

# hyperparameters
learning_rate = 0.01
num_epochs = 10
batch_size = 512  # parameter of DataLoader

# step 1: load data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        os.path.join(data_path, 'data_mnist'), train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))  # best
        ])
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        os.path.join(data_path, 'data_mnist'), train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    ),
    batch_size=batch_size,
    shuffle=False,
)

# Take out the sample for direct observation
x, y = next(iter(train_loader))
print(x.shape, y.shape)  # torch.Size([512, 1, 28, 28]) torch.Size([512])
print(x.min(), x.max())  # tensor(-0.4242) tensor(2.8215)
plot_image(x, y, "image sample")


# step 2: define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)  # image shape 28*28
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes

    def forward(self, x):
        # x.shape(batch_size, 1, 28, 28)
        x = F.relu(self.fc1(x))  # (batch_size, 256)
        x = F.relu(self.fc2(x))  # (batch_size, 64)
        x = self.fc3(x)  # (batch_size, 10)
        return x


# step 3: train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # cuda
net = Net().to(device)  # Instantiate the model on the device

optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # Optimization model parameters
train_loss = []  # record the loss
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}")

    for batch_idx, (x, y) in enumerate(train_loader):
        # x.shape(batch_size, 1, 28, 28)
        # y.shape(batch_size)
        x = x.view(x.size(0), 28 * 28).to(device)  # (batch_size, 784)
        out = net(x)  # (batch_size, 10)
        y_onehot = one_hot(y, 10).to(device)  # (batch_size, 10)

        loss = F.mse_loss(out, y_onehot)
        optimizer.zero_grad()  # clear the gradients for this training step
        loss.backward()  # calculate the gradients
        optimizer.step()  # update the parameters

        train_loss.append(loss.item())  # record the loss

        if batch_idx % 10 == 0:
            print("Train Epoch: {epoch} [{completed}/{total}] ({percent:.0f}%)\tLoss: {loss:.6f}".format(
                epoch=epoch + 1,
                completed=batch_idx * len(x),
                total=len(train_loader.dataset),
                percent=100. * batch_idx / len(train_loader),
                loss=loss.item()
            ))

        """
        Epoch 1
        Train Epoch: 1 [0/60000 (0%)]	Loss: 0.115833
        Train Epoch: 1 [5120/60000 (8%)]	Loss: 0.095876
        Train Epoch: 1 [10240/60000 (17%)]	Loss: 0.083277
        Train Epoch: 1 [15360/60000 (25%)]	Loss: 0.077820
        Train Epoch: 1 [20480/60000 (34%)]	Loss: 0.072549
        Train Epoch: 1 [25600/60000 (42%)]	Loss: 0.066049
        Train Epoch: 1 [30720/60000 (51%)]	Loss: 0.064271
        Train Epoch: 1 [35840/60000 (59%)]	Loss: 0.060865
        Train Epoch: 1 [40960/60000 (68%)]	Loss: 0.060129
        Train Epoch: 1 [46080/60000 (76%)]	Loss: 0.054560
        Train Epoch: 1 [51200/60000 (85%)]	Loss: 0.051744
        Train Epoch: 1 [56320/60000 (93%)]	Loss: 0.049708
        Epoch 2
        Train Epoch: 2 [0/60000 (0%)]	Loss: 0.050642
        Train Epoch: 2 [5120/60000 (8%)]	Loss: 0.048092
        Train Epoch: 2 [10240/60000 (17%)]	Loss: 0.045999
        Train Epoch: 2 [15360/60000 (25%)]	Loss: 0.044162
        Train Epoch: 2 [20480/60000 (34%)]	Loss: 0.045289
        Train Epoch: 2 [25600/60000 (42%)]	Loss: 0.042404
        Train Epoch: 2 [30720/60000 (51%)]	Loss: 0.042469
        Train Epoch: 2 [35840/60000 (59%)]	Loss: 0.040225
        Train Epoch: 2 [40960/60000 (68%)]	Loss: 0.039921
        Train Epoch: 2 [46080/60000 (76%)]	Loss: 0.041059
        Train Epoch: 2 [51200/60000 (85%)]	Loss: 0.040481
        Train Epoch: 2 [56320/60000 (93%)]	Loss: 0.036657
        ...
        """

plot_curve(train_loss)  # plot the loss curve after recording the loss
# we get optimal [w1,w2,w3, b1,b2,b3] after training

# step 4: test the model
total_num = len(test_loader.dataset)
total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28).to(device)  # (batch_size, 784)
    y = y.to(device)  # (batch_size)

    out = net(x)  # (batch_size, 10)
    pred = out.argmax(dim=1)  # (batch_size, 1)

    correct = pred.eq(y).sum().float().item()
    total_correct += correct

accuracy = total_correct / total_num
print("\ntotal num: {}, total correct: {}, accuracy: {:.2f}%".format(
    total_num, total_correct, accuracy * 100))

# Take out the sample for direct observation
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), 28 * 28).to(device))  # (batch_size, 10)
pred = out.argmax(dim=1)  # (batch_size, 1)
plot_image(x, pred, "test")  # plot the predicted label
