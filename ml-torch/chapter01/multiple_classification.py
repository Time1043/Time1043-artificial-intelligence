import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn

# hyper-parameters
batch_size = 64  # for data loader
input_size = 28 * 28
output_size = 10
num_epochs = 10  # for train

# check gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data
transformation = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='data/mnist', train=True, download=True, transform=transformation)
test_dataset = torchvision.datasets.MNIST(root='data/mnist', train=False, download=True, transform=transformation)
# data loader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# check samples
for i, (images, labels) in enumerate(train_dataloader):
    print(images.shape, labels.shape)

    plt.imshow(images[0][0], cmap='gray')
    plt.show()
    print(labels[0])

    if i > 10:
        break


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        logics = self.linear(x)
        return logics


model = Model(input_size, output_size).to(device)

# train and test
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.view(-1, input_size).to(device)
            y = y.to(device)

            logics = model(x)
            _, predicted = torch.max(logics.data, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.long().to(device)

        # forward propagation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward propagation and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = evaluate(model, test_dataloader)
    print(f'Epoch {epoch + 1}: test accuracy = {accuracy:.2f}')

"""
Epoch 1: test accuracy = 0.87
Epoch 2: test accuracy = 0.88
Epoch 3: test accuracy = 0.89
Epoch 4: test accuracy = 0.90
Epoch 5: test accuracy = 0.90
Epoch 6: test accuracy = 0.90
Epoch 7: test accuracy = 0.90
Epoch 8: test accuracy = 0.91
Epoch 9: test accuracy = 0.91
Epoch 10: test accuracy = 0.91
"""
