from learner import Learner
from models.lenet import LeNet
from models.resnet import ResNet18

if __name__ == '__main__':
    data_dir = '~/Dataset'
    learner = Learner(data_dir, ResNet18)
    learner.train()
    learner.test()
