import torch
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# hyper-parameters
batch_size = 100  # for data loader

input_size = 28 * 28  # 输入大小
hidden_size = 512  # 隐藏层大小
num_classes = 10  # 输出大小 (classifications)

learning_rate = 0.001  # for optimizer
num_epochs = 10  # for training

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data
train_data = datasets.MNIST(root="data/mnist", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="data/mnist", train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        """
        init MLP
        :param input_size: 输入数据的维度
        :param hidden_size: 隐藏层大小
        :param num_classes: 输出大小 (classifications)
        """

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        forward propagation
        :param x: 输入数据
        :return: 输出数据
        """

        out = self.fc1(x)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        return out


# Instantiate the MLP model
model = MLP(input_size, hidden_size, num_classes).to(device)

# train (loss, optimizer)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)  # images -> vector
        labels = labels.to(device)

        # forward propagation
        outputs = model(images)  # data to model
        loss = criterion(outputs, labels)

        # backward propagation and update parameters
        optimizer.zero_grad()  # clear gradients
        loss.backward()
        optimizer.step()

        # print loss every 100 steps
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# test
with torch.no_grad():
    correct = 0
    total = 0

    # from test_loader get images and labels
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)  # images -> vector
        labels = labels.to(device)

        # forward propagation
        outputs = model(images)  # data to model
        _, predicted = torch.max(outputs.data, 1)  # predicted

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print accuracy
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

# save
torch.save(model, "output/mnist_mlp_model.pkl")

"""
cuda
Epoch [1/10], Step [100/600], Loss: 0.2783
Epoch [1/10], Step [200/600], Loss: 0.3115
Epoch [1/10], Step [300/600], Loss: 0.1504
Epoch [1/10], Step [400/600], Loss: 0.0884
Epoch [1/10], Step [500/600], Loss: 0.1849
Epoch [1/10], Step [600/600], Loss: 0.0996
Epoch [2/10], Step [100/600], Loss: 0.1644
Epoch [2/10], Step [200/600], Loss: 0.0943
Epoch [2/10], Step [300/600], Loss: 0.1490
Epoch [2/10], Step [400/600], Loss: 0.0732
Epoch [2/10], Step [500/600], Loss: 0.0564
Epoch [2/10], Step [600/600], Loss: 0.0546
Epoch [3/10], Step [100/600], Loss: 0.0328
Epoch [3/10], Step [200/600], Loss: 0.0418
Epoch [3/10], Step [300/600], Loss: 0.0170
Epoch [3/10], Step [400/600], Loss: 0.0881
Epoch [3/10], Step [500/600], Loss: 0.0326
Epoch [3/10], Step [600/600], Loss: 0.0507
Epoch [4/10], Step [100/600], Loss: 0.0373
Epoch [4/10], Step [200/600], Loss: 0.0533
Epoch [4/10], Step [300/600], Loss: 0.0239
Epoch [4/10], Step [400/600], Loss: 0.0195
Epoch [4/10], Step [500/600], Loss: 0.0348
Epoch [4/10], Step [600/600], Loss: 0.0243
Epoch [5/10], Step [100/600], Loss: 0.0169
Epoch [5/10], Step [200/600], Loss: 0.0208
Epoch [5/10], Step [300/600], Loss: 0.0332
Epoch [5/10], Step [400/600], Loss: 0.0035
Epoch [5/10], Step [500/600], Loss: 0.0308
Epoch [5/10], Step [600/600], Loss: 0.0557
Epoch [6/10], Step [100/600], Loss: 0.0076
Epoch [6/10], Step [200/600], Loss: 0.0045
Epoch [6/10], Step [300/600], Loss: 0.0217
Epoch [6/10], Step [400/600], Loss: 0.1173
Epoch [6/10], Step [500/600], Loss: 0.0122
Epoch [6/10], Step [600/600], Loss: 0.0264
Epoch [7/10], Step [100/600], Loss: 0.0294
Epoch [7/10], Step [200/600], Loss: 0.0173
Epoch [7/10], Step [300/600], Loss: 0.0100
Epoch [7/10], Step [400/600], Loss: 0.0099
Epoch [7/10], Step [500/600], Loss: 0.1083
Epoch [7/10], Step [600/600], Loss: 0.0164
Epoch [8/10], Step [100/600], Loss: 0.0895
Epoch [8/10], Step [200/600], Loss: 0.0031
Epoch [8/10], Step [300/600], Loss: 0.0056
Epoch [8/10], Step [400/600], Loss: 0.0412
Epoch [8/10], Step [500/600], Loss: 0.0095
Epoch [8/10], Step [600/600], Loss: 0.0517
Epoch [9/10], Step [100/600], Loss: 0.0036
Epoch [9/10], Step [200/600], Loss: 0.0097
Epoch [9/10], Step [300/600], Loss: 0.0131
Epoch [9/10], Step [400/600], Loss: 0.0086
Epoch [9/10], Step [500/600], Loss: 0.0060
Epoch [9/10], Step [600/600], Loss: 0.0037
Epoch [10/10], Step [100/600], Loss: 0.0124
Epoch [10/10], Step [200/600], Loss: 0.0240
Epoch [10/10], Step [300/600], Loss: 0.0859
Epoch [10/10], Step [400/600], Loss: 0.0043
Epoch [10/10], Step [500/600], Loss: 0.0114
Epoch [10/10], Step [600/600], Loss: 0.0011
Accuracy of the network on the 10000 test images: 98.11 %
"""