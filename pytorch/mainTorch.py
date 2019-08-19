import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 32

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def instance(dataloader, classes, batch_size=BATCH_SIZE):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join([classes[labels[j]] for j in range(batch_size)]))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Learner():
    def __init__(self, data_dir):
        self.net = Net()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.opt = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9)

    def load_data(self, data_dir):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                transform=transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                transform=transform)
        self.testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                shuffle=True, num_workers=2)

    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = nn.CrossEntropyLoss(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def train(self, n_epoch=10):
        for epoch in range(n_epoch):
            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.opt.zero_grad()

                outputs = self.net(inputs)
                accuracy, loss = self.metrics(outputs, labels)
                loss.backward()
                self.opt.step()

                running_loss += loss
                if (i+1) % 1000 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/1000))
                    print(accuracy)
                    running_loss = 0

        print('Finished Training')

#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    resnet = ResNet()
    resnet.train()
