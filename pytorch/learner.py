import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.lenet import Net

BATCH_SIZE = 32
INIT_LR = 1e-2
MOMENTUM = 0.9
L2_REG = 5e-4
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

class Learner():
    def __init__(self, data_dir):
        # set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # set data loader pipeline
        self.trainloader = self._build_dataloader(data_dir, train=True)
        self.testloader = self._build_dataloader(data_dir, train=False)

        # define forward computational graph
        self.net = Net()
        self.net.to(self.device)

        # setup loss function
        self.criterion = self.loss_fn()

        # setup optimizatizer
        self.opt = self.setup_optimizer()
        self.lr_scheduler = self.setup_lr_scheduler(self.opt)

    def _build_dataloader(self, data_dir, train=True):
        transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=4)
        return dataloader

    def loss_fn(self):
        return nn.CrossEntropyLoss()

    def setup_optimizer(self):
        return optim.SGD(self.net.parameters(), lr=INIT_LR, momentum=MOMENTUM)
                #weight_decay=L2_REG)

    def setup_lr_scheduler(self, opt):
        return torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 30], gamma=0.1)

    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def train(self, n_epoch=10):
        for epoch in range(n_epoch):
            self.lr_scheduler.step()
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.opt.zero_grad()

                outputs = self.net(inputs)
                accuracy, loss = self.metrics(outputs, labels)
                loss.backward()
                self.opt.step()

                if (i+1) % 100 == 0:
                    print(epoch+1, ' acc.={}, loss={}'.format(accuracy, loss))
            self.test()

        print('Finished Training')

    def test(self):
        total_accuracy_sum = 0
        total_loss_sum = 0
        for i, data in enumerate(self.testloader, 0):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.net(images)
            accuracy, loss = self.metrics(outputs, labels)
            total_accuracy_sum += accuracy
            total_loss_sum += loss
        avg_loss = total_loss_sum / len(self.testloader)
        avg_acc = total_accuracy_sum / len(self.testloader)
        print('acc.= {0}, loss={1}'.format(avg_acc, avg_loss))
        print('test')