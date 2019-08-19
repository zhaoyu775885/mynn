import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 32
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # set data loader pipeline
        self.trainloader = self._build_dataloader(data_dir, train=True)
        self.testloader = self._build_dataloader(data_dir, train=False)

        self.net = Net()
        self.net.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.opt = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9)

    def _build_dataloader(self, data_dir, train=True):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=2)
        return dataloader

    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def train(self, n_epoch=10):
        for epoch in range(n_epoch):
            running_loss = 0
            for i, data in enumerate(self.trainloader, 0):
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

if __name__ == '__main__':
    data_dir = '~/Dataset'
    learner = Learner(data_dir)
    learner.train()
