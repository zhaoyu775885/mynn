import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from learner.abstract_learner import AbstractLearner

BATCH_SIZE = 128
INIT_LR = 1e-1
MOMENTUM = 0.9
L2_REG = 5e-4


class FullLearner(AbstractLearner):
    def __init__(self, dataset, net, device='cuda:1', log_path='./log', teacher=None):
        super(FullLearner, self).__init__(dataset, net, device, log_path, teacher)

        self.train_loader = self._build_dataloader(BATCH_SIZE, is_train=True)  # batch_size should be
        self.test_loader = self._build_dataloader(100, is_train=False)  # parameterized

        # setup optimizer
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    def _loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer(self):
        return optim.SGD(self.forward.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)

    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[100, 150, 200], gamma=0.1)

    def metrics(self, logits, labels, kd=None):
        _, predicted = torch.max(logits, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(logits, labels)
        if kd is not None:
            loss += self.teacher.kd_loss(logits, kd)
        accuracy = correct / labels.size(0)
        return accuracy, loss

    def cnt_flops(self):
        flops = 0
        for i, data in enumerate(self.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            flops = self.forward.cnt_flops(inputs)
            break
        return flops

    def train(self, n_epoch=40):
        self.net.train()
        for epoch in range(n_epoch):

            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'accuracy': 0, 'lr': self.opt.param_groups[0]['lr']})

            for i, data in enumerate(self.train_loader):

                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                logits = self.forward(inputs)
                kd = None if self.teacher is None else self.teacher.infer(inputs)
                accuracy, loss = self.metrics(logits, labels, kd=kd)
                self.recoder.add_info(labels.size(0), {'loss': loss, 'accuracy': accuracy})

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * BATCH_SIZE / time_step)
                    print(i + 1, ': lr={0:.1e} | acc={1: 5.2f} | loss={2:5.3f} | speed={3} pic/s'.format(
                        self.opt.param_groups[0]['lr'], accuracy * 100, loss, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler.step()

            if (epoch + 1) % 10 == 0:
                self.save_model()
                self.test()
                self.net.train()
        print('Finished Training')

    def test(self):
        self.net.eval()
        total_accuracy_sum, total_loss_sum = 0, 0
        for i, data in enumerate(self.test_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            logits = self.forward(inputs)
            accuracy, loss = self.metrics(logits, labels)
            total_accuracy_sum += accuracy
            total_loss_sum += loss.item()
        avg_loss = total_loss_sum / len(self.test_loader)
        avg_acc = total_accuracy_sum / len(self.test_loader)
        print('val:\n, accuracy={0:.2f}%, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))
