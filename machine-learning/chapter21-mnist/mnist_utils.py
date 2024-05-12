import matplotlib.pyplot as plt
import torch


def plot_curve(data):
    """ Plot the curve of data """
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(["value"], loc="upper left")
    plt.xlabel("step")
    plt.ylabel("value")
    plt.show()


def plot_image(img, label, name):
    """ Plot the image with label """
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, num_classes):
    """ Convert label to one-hot vector """
    # return torch.zeros(len(label), num_classes).scatter_(1, label.unsqueeze(1), 1)
    out = torch.zeros(label.size(0), num_classes)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(1, idx, 1)
    return out
