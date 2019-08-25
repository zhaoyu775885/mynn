import torch
import torch.nn as nn
import torch.optim as optim

from dataset.cifar10 import Cifar10
from models.lenet import LeNet

BATCH_SIZE = 128
INIT_LR = 1e-2
MOMENTUM = 0.9
L2_REG = 5e-4
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img/2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# def instance(dataloader, classes, batch_size=BATCH_SIZE):
#     dataiter = iter(dataloader)
#     images, labels = dataiter.next()
#     imshow(torchvision.utils.make_grid(images))
#     print(' '.join([classes[labels[j]] for j in range(batch_size)]))

class Learner():
    def __init__(self, data_dir, net=LeNet):
        # set device & build dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = Cifar10(data_dir)

        # build dataloader
        self.trainloader = self._build_dataloader(BATCH_SIZE, is_train=True)
        self.testloader = self._build_dataloader(100, is_train=False)

        # define forward computational graph
        self.forward = net().to(self.device)

        # setup loss function
        self.criterion = self._loss_fn()

        # setup optimizatizer
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    # def _forward(self):
    #     net = LeNet()
    #     return net.to(self.device)

    def _build_dataloader(self, batch_size, is_train):
        return self.dataset.build_dataloader(batch_size, is_train)

    def _loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        return optim.SGD(self.forward.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)

    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[100, 150, 200], gamma=0.1)

    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def train(self, n_epoch=250):
        for epoch in range(n_epoch):
            self.lr_scheduler.step()
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.opt.zero_grad()

                outputs = self.forward(inputs)
                accuracy, loss = self.metrics(outputs, labels)
                loss.backward()
                self.opt.step()

                if (i+1) % 100 == 0:
                    print(i+1, ' acc={0:.2f}, loss={1:.3f}'.format(accuracy*100, loss))
            print(epoch+1, 'finished')
            self.test()

        print('Finished Training')

    def test(self):
        total_accuracy_sum = 0
        total_loss_sum = 0
        for i, data in enumerate(self.testloader, 0):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs = self.forward(images)
            accuracy, loss = self.metrics(outputs, labels)
            total_accuracy_sum += accuracy
            total_loss_sum += loss.item()
        avg_loss = total_loss_sum / len(self.testloader)
        avg_acc = total_accuracy_sum / len(self.testloader)
        print('acc= {0:.2f}, loss={1:.3f}'.format(avg_acc*100, avg_loss))