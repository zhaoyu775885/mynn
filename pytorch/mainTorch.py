from learner import Learner
from models.lenet import LeNet
from models.resnet import ResNet20

if __name__ == '__main__':
    data_dir = '~/Dataset'
    learner = Learner(data_dir, ResNet20)
    learner.train()
#    learner.test()
