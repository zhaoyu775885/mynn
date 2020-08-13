from datasets.cifar import Cifar10, Cifar100
from nets.resnet import ResNet20, ResNet32
from nets.resnet_lite_v2 import ResNet20Lite
from learner.learner import Learner
from learner.learner_dcps import DLearner

if __name__ == '__main__':
    # specify dataset
    cifar10 = '/home/zhaoyu/Datasets/cifar10'
    cifar100 = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(cifar100)
    # dataset = Cifar10(cifar10)

    # specify network model
    # net = ResNet20(n_classes=100)
    net = ResNet20Lite(n_classes=100)

    # init Leaner
    learner = DLearner(dataset, net)

    learner.train(n_epoch=200)
    learner.load_model()
    learner.test()
