from datasets.cifar import Cifar10, Cifar100

from nets.resnet import ResNet20, ResNet32
from nets.resnet_lite import ResNet20Lite, ResNet32Lite, ResNet56Lite
from nets.resnet_gated import ResNet20Gated, ResNet32Gated, ResNet56Gated

from learner.prune import DcpsLearner
from learner.full import FullLearner
from learner.distiller import Distiller

if __name__ == '__main__':
    cifar100_flag = True
    prune_flag = False
    lite_flag = True

    # specify dataset
    if not cifar100_flag:
        cifar10_path = '/home/zhaoyu/Datasets/cifar10'
        dataset = Cifar10(cifar10_path)
        n_class = 10
    else:
        cifar100_path = '/home/zhaoyu/Datasets/cifar100'
        dataset = Cifar100(cifar100_path)
        n_class = 100

    device = 'cuda:0'
    teacher_net = ResNet20(n_classes=n_class)
    teacher = Distiller(dataset, teacher_net, device=device, model_path='./models/6884.pth')
    if not prune_flag:
        if not lite_flag:
            net = ResNet20(n_classes=n_class)
        else:
            net = ResNet20Lite(n_classes=n_class)
        learner = FullLearner(dataset, net, device=device, teacher=teacher)
        print(learner.cnt_flops())
    else:
        net = ResNet20Gated(n_classes=dataset.n_class)
        learner = DcpsLearner(dataset, net, device=device)

    save_path = './models/full/model.pth'
    learner.train(save_path=save_path)
    learner.load_model(path=save_path)
    learner.test()
