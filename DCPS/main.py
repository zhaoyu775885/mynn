from datasets.cifar import Cifar100
from nets.resnet import ResNet20
from learner.learner import Learner

if __name__ == '__main__':
    # specify dataset
    cifar10 = '/home/zhaoyu/Datasets/cifar10'
    cifar100 = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(cifar100)

    # specify network model
    net = ResNet20(n_classes=100)
    # net = resnet20()
    # net = Lenet()

    # init Leaner
    learner = Learner(dataset, net)

    learner.train(n_epoch=200)
    learner.load_model()
    learner.test()
