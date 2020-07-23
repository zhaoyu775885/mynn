from datasets.cifar10 import Cifar10
from nets.resnet import ResNet20
from nets.resnet_std import resnet20
from nets.lenet import Lenet
from learner import Learner

if __name__ == '__main__':
    # specify dataset
    data_path = '/home/zhaoyu/Datasets/cifar10'
    dataset = Cifar10(data_path)

    # specify network model
    net = ResNet20()
    # net = resnet20()
    # net = Lenet()

    # init Leaner
    leaner = Learner(dataset, net)
    print(': lr={2:.1e} | acc={0: 5.2f} | loss={1:5.3f} | speed={3} pic/s'.format(
        1e-3, 10, 0.001, 5000))
    exit(1)

    # leaner.train(n_epoch=200)
    leaner.test()
