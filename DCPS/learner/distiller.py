import torch
import torch.nn as nn
from datasets.cifar import Cifar10, Cifar100
from nets.resnet import ResNet20
from learner.full import FullLearner

BATCH_SIZE = 128
INIT_LR = 1e-1

class Distiller(FullLearner):
    def __init__(self, dataset, net, device='cuda:0', model_path='./models/6884.pth'):
        super(Distiller, self).__init__(dataset, net, device)
        self.load_model(model_path)
        self.kd_loss = nn.KLDivLoss(reduction='batchmean')
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.net.eval()

    def infer(self, images):
        return self.net(images).detach()

    def loss(self, std_logits, trg_logits):
        T = 4
        log_prob = self.logsoftmax(std_logits/T)
        prob = self.softmax(trg_logits/T)
        w_dst = 4
        loss = self.kd_loss(log_prob, prob) * w_dst
        return loss

if __name__ == '__main__':
    cifar100 = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(cifar100)

    net = ResNet20(n_classes=100)
    learner = Distiller(dataset, net)

    learner.test()