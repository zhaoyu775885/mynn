import torch.nn as nn
from learner.full import FullLearner


class Distiller(FullLearner):
    def __init__(self, dataset, net, device='cuda:0', model_path='./models/6884.pth'):
        super(Distiller, self).__init__(dataset, net, device)
        ''' handle exceptions '''
        self.load_model(model_path)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.net.eval()

    def infer(self, images):
        return self.net(images).detach()

    def kd_loss(self, std_logits, trg_logits):
        T = 4
        log_prob = self.log_softmax(std_logits / T)
        prob = self.softmax(trg_logits / T)
        w_dst = 4
        loss = self.loss(log_prob, prob) * w_dst
        return loss


if __name__ == '__main__':
    from datasets.cifar import Cifar100
    from nets.resnet import ResNet20

    cifar100_path = '/home/zhaoyu/Datasets/cifar100'
    cifar100 = Cifar100(cifar100_path)

    net = ResNet20(n_classes=100)
    learner = Distiller(cifar100, net)

    learner.test()
