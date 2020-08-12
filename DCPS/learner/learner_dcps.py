import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 128
INIT_LR = 1e-1
MOMENTUM = 0.9
L2_REG = 5e-4

class DLearner():
    def __init__(self, Dataset, Net):
        # set device & build dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.dataset = Dataset
        self.net = Net

        # build dataloader
        self.trainloader = self._build_dataloader(BATCH_SIZE, is_train=True)
        self.testloader = self._build_dataloader(100, is_train=False)

        # define forward computational
        self.forward = self.net.to(self.device)

        # setup loss function
        self.criterion = self._loss_fn()

        # setup optimizatizer
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

        #setup writer
        self.writer = SummaryWriter('./log/')

    def _build_dataloader(self, batch_size, is_train):
        return self.dataset.build_dataloader(batch_size, is_train)

    def _loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        return optim.SGD(self.forward.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)

    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[80, 120, 160], gamma=0.1)
        # return torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, )

    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def display_perf(self, acc, loss):
        '''future utility'''
        pass

    def train(self, n_epoch=40):
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            # batch training within each epoch
            time_prev = timer()
            epoch_loss = 0
            epoch_accuracy = 0
            epoch_samp_count = 0
            epoch_lr = 0
            for i, data in enumerate(self.trainloader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.forward(inputs)
                accuracy, loss = self.metrics(outputs, labels)
                epoch_loss += loss * labels.size(0)
                epoch_accuracy += accuracy * labels.size(0)
                epoch_samp_count += labels.size(0)
                epoch_lr = self.opt.param_groups[0]['lr']

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100*BATCH_SIZE/time_step)
                    print(i+1, ': lr={0:.1e} | acc={1: 5.2f} | loss={2:5.3f} | speed={3} pic/s'.format(
                        epoch_lr, accuracy * 100, loss, speed))
                    time_prev = timer()

            self.writer.add_scalar('Train/lr', epoch_lr, epoch)
            self.writer.add_scalar('Train/loss', epoch_loss/epoch_samp_count, epoch)
            self.writer.add_scalar('Train/acc.', epoch_accuracy/epoch_samp_count, epoch)
            self.writer.flush()

            self.lr_scheduler.step()
            if (epoch+1) % 10 == 0:
                self.save_model()
                self.test()
                self.net.train()
        print('Finished Training')

    def test(self):
        self.net.eval()
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
        print('acc= {0:.2f}, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))

    def save_model(self, path='./models/models.pth'):
        # todo: supplement the epoch info
        torch.save(self.net.state_dict(), path)

    def load_model(self, path='./models/models.pth'):
        """
        make sure that the checkpoint on the disk contains all related variables
        in current network.
        """
        disk_state_dict = torch.load(path)
        try:
            self.net.load_state_dict(disk_state_dict)
        except RuntimeError:
            print('Dismatched models, please check the network.')
            state_dict = self.net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = disk_state_dict[key]
            self.net.load_state_dict(state_dict)
