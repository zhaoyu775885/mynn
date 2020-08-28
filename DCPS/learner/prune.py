import torch
import torch.nn as nn
import torch.optim as optim
from timeit import default_timer as timer
from nets.resnet_lite import ResNetLite
from nets.resnet import ResNet20, ResNet32
import dcps.DNAS as dnas
from learner.abstract_learner import AbstractLearner
from learner.full import FullLearner
from learner.distiller import Distiller

BATCH_SIZE = 128
INIT_LR = 1e-1
MOMENTUM = 0.9
L2_REG = 5e-4


class DcpsLearner(AbstractLearner):
    def __init__(self, dataset, net, device='cuda:1', log_path='./log', teacher=None):
        super(DcpsLearner, self).__init__(dataset, net, device, log_path, teacher)

        self.train_loader = self._build_dataloader(BATCH_SIZE, is_train=True)
        self.test_loader = self._build_dataloader(100, is_train=False)

        # setup optimizer
        self.opt_warmup = self._setup_optimizer_warmup()
        self.lr_scheduler_warmup = self._setup_lr_scheduler_warmup()

        self.opt_train = self._setup_optimizer_train()
        self.lr_scheduler_train = self._setup_lr_scheduler_train()

        self.opt_search = self._setup_optimizer_search()
        self.lr_scheduler_search = self._setup_lr_scheduler_search()

    def _loss_fn(self):
        return nn.CrossEntropyLoss()

    def _setup_optimizer_warmup(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)

    def _setup_optimizer_train(self):
        vars = [item[1] for item in self.forward.named_parameters() if 'gate' not in item[0]]
        return optim.SGD(vars, lr=INIT_LR * 0.1, momentum=MOMENTUM, weight_decay=L2_REG)

    def _setup_optimizer_search(self):
        # todo: the risk that ignores the shared gates in the network
        gates = [item[1] for item in self.forward.named_parameters() if 'gate' in item[0]]
        return optim.Adam(gates, lr=0.01)

    def _setup_lr_scheduler_warmup(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_warmup, milestones=[100, 150], gamma=0.1)

    def _setup_lr_scheduler_train(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_train, milestones=[50, 100], gamma=0.1)

    def _setup_lr_scheduler_search(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt_search, milestones=[50, 100], gamma=0.1)

    def metrics(self, outputs, labels, flops=None):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        loss_with_flops = loss + 3 * torch.log(flops)
        accuracy = correct / labels.size(0)
        return accuracy, loss, loss_with_flops

    def train(self, n_epoch=250):
        self.train_warmup(n_epoch=150, save_path='./models/warmup/model.pth')
        tau = self.train_search(n_epoch=200,
                                load_path='./models/warmup/model.pth',
                                save_path='./models/search/model.pth')
        self.train_prune(tau=tau, n_epoch=n_epoch,
                         load_path='./models/search/model.pth',
                         save_path='./models/prune/model.pth')

    def train_warmup(self, n_epoch=200, save_path='./models/warmup/model.pth'):
        self.net.train()
        for epoch in range(n_epoch):
            print('epoch: ', epoch + 1)
            time_prev = timer()
            self.recoder.init({'loss': 0, 'accuracy': 0, 'lr': self.opt_warmup.param_groups[0]['lr']})
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs, flops, flops_list = self.forward(inputs)
                accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                self.recoder.add_info(labels.size(0), {'loss': loss, 'accuracy': accuracy})
                self.opt_warmup.zero_grad()
                loss.backward()
                self.opt_warmup.step()
                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * BATCH_SIZE / time_step)
                    print(i + 1, ': lr={0:.1e} | acc={1: 5.2f} | loss={2:5.3f} | flops={3} | speed={4} pic/s'.format(
                        self.opt_warmup.param_groups[0]['lr'], accuracy * 100, loss, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_warmup.step()
            if (epoch + 1) % 10 == 0:
                self.save_model(path=save_path)
                self.test_dcps()
                self.net.train()
        print('Finished Warming-up')

    def train_search(self, n_epoch=200,
                     load_path='./models/warmup/model.pth',
                     save_path='./models/search/model.pth'):
        self.load_model(path=load_path)
        tau = 1.0
        for epoch in range(n_epoch):
            time_prev = timer()
            tau = 10 ** (1 - 2 * epoch / (n_epoch - 1))
            print('epoch: ', epoch + 1, ' tau: ', tau)
            self.recoder.init({'loss': 0, 'loss_f': 0, 'accuracy': 0,
                               'lr': self.opt_train.param_groups[0]['lr'],
                               'tau': tau})
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                if i % 2:
                    self.net.eval()
                    outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
                    accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                    self.opt_search.zero_grad()
                    loss_with_flops.backward()
                    self.opt_search.step()
                else:
                    self.net.train()
                    outputs, _a, flops, _b = self.forward(inputs, tau=tau, noise=True)
                    accuracy, loss, loss_with_flops = self.metrics(outputs, labels, flops)
                    self.opt_train.zero_grad()
                    loss.backward()
                    self.opt_train.step()
                self.recoder.add_info(labels.size(0), {'loss': loss, 'loss_f': loss_with_flops,
                                                       'accuracy': accuracy})
                if (i + 1) % 100 == 0:
                    time_step = timer() - time_prev
                    speed = int(100 * BATCH_SIZE / time_step)
                    print(i + 1,
                          ': lr={0:.1e} | acc={1: 5.2f} |'.format(self.opt_train.param_groups[0]['lr'], accuracy * 100),
                          'loss={0:5.3f} | loss_f={1:5.3f} | flops={2} | speed={3} pic/s'.format(
                              loss, loss_with_flops, flops, speed))
                    time_prev = timer()
            self.recoder.update(epoch)
            self.lr_scheduler_train.step()
            self.lr_scheduler_search.step()
            if (epoch + 1) % 10 == 0:
                self.save_model(path=save_path)
                self.test_dcps(tau=tau)
                display_info(flops_list, prob_list)
        print('Finished Training')
        return tau

    def train_prune(self, tau, n_epoch=250,
                    load_path='./models/search/model.pth',
                    save_path='./models/prune/model.pth'):
        # Done, 0. load the searched model and extract the prune info
        # Done, 1. define the pure small network based on prune info
        # Done, 2. train and validate, exploit the full learner?
        self.load_model(load_path)
        dcfg = dnas.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
        channel_list_20 = [16, [[16, 16], [16, 16], [16, 16]], [[32, 32, 32], [32, 32], [32, 32]],
                           [[64, 64, 64], [64, 64], [64, 64]]]

        data = next(iter(self.train_loader))
        inputs, labels = data[0].to(self.device), data[1].to(self.device)
        self.net.eval()
        outputs, prob_list, flops, flops_list = self.forward(inputs, tau=tau, noise=False)
        channel_list_prune = get_prune_list(channel_list_20, prob_list, dcfg=dcfg)
        print(channel_list_prune)
        # channel_list_prune = [13, [[9, 13], [9, 13], [16, 13]], [[16, 25, 25], [27, 25], [16, 25]], [[54, 64, 64], [45, 64], [29, 64]]]

        teacher_net = ResNet20(n_classes=self.dataset.n_class)
        teacher = Distiller(self.dataset, teacher_net, device=self.device, model_path='./models/6884.pth')
        net = ResNetLite(20, self.dataset.n_class, channel_list_prune)
        full_learner = FullLearner(self.dataset, net, device=self.device, teacher=teacher)
        print(full_learner.cnt_flops())
        full_learner.train(n_epoch=n_epoch)

    def test(self, tau):
        self.net.eval()
        total_accuracy_sum = 0
        total_loss_sum = 0
        for i, data in enumerate(self.test_loader, 0):
            images, labels = data[0].to(self.device), data[1].to(self.device)
            outputs, prob_list, flops, flops_list = self.forward(images, tau=tau, noise=False)
            accuracy, loss, _ = self.metrics(outputs, labels, flops)
            total_accuracy_sum += accuracy
            total_loss_sum += loss.item()
        avg_loss = total_loss_sum / len(self.test_loader)
        avg_acc = total_accuracy_sum / len(self.test_loader)
        print('acc= {0:.2f}, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))


def get_prune_list(resnet_channel_list, prob_list, dcfg):
    import numpy as np
    prune_list = []
    idx = 0

    chn_input_full, chn_output_full = 3, resnet_channel_list[0]
    dnas_conv = lambda input, output: dnas.Conv2d(input, output, 1, 1, 1, False, dcfg=dcfg)
    conv = dnas_conv(chn_input_full, chn_output_full)
    chn_output_prune = int(np.ceil(
        min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
    ))
    prune_list.append(chn_output_prune)
    chn_input_full = chn_output_full
    idx += 1
    for blocks in resnet_channel_list[1:]:
        blocks_list = []
        for block in blocks:
            block_prune_list = []
            for chn_output_full in block:
                conv = dnas.Conv2d(chn_input_full, chn_output_full, 1, 1, 1, False, dcfg=dcfg)
                chn_output_prune = int(np.ceil(
                    min(torch.dot(prob_list[idx], conv.out_plane_list).item(), chn_output_full)
                ))
                block_prune_list.append(chn_output_prune)
                chn_input_full = chn_output_full
                idx += 1
            blocks_list.append(block_prune_list)
        prune_list.append(blocks_list)
    return prune_list


def display_info(flops_list, prob_list):
    print('=============')
    for flops, prob in zip(flops_list, prob_list):
        pinfo = ''
        for item in prob.tolist():
            pinfo += '{0:.3f} '.format(item)
        pinfo += ', {0:.0f}'.format(flops)
        print(pinfo)
    print('-------------\n')
