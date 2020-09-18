import os
import sys
import argparse
from datasets.cifar import Cifar10, Cifar100
from nets.resnet import ResNet, ResNet20, ResNet32
from nets.resnet_lite import ResNet20Lite, ResNet32Lite, ResNet56Lite
from nets.resnet_gated import ResNetGated
from learner.prune import DcpsLearner
from learner.full import FullLearner
from learner.distiller import Distiller

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# prepare for parallel training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'], help='Dataset Name')
    parser.add_argument('--data_path', type=str, help='Dataset Directory')
    parser.add_argument('--net', default='resnet', choices=['resnet', 'mobilenet'], help='Net')
    parser.add_argument('--net_index', default=20, type=int, choices=[20, 32], help='Index')
    parser.add_argument('--num_epoch', default=250, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch Size')
    parser.add_argument('--std_batch_size', default=128, type=int, help='Norm Batch Size')
    parser.add_argument('--std_init_lr', default=1e-1, type=float, help='Norm Init Lr')
    parser.add_argument('--dst_flag', default=0, type=int, help='Dst Flag')
    parser.add_argument('--prune_flag', default=0, type=int, help='Prune Flag')
    parser.add_argument('--teacher_net', default='resnet', choices=['resnet'], help='Net')
    parser.add_argument('--teacher_net_index', default=20, type=int, choices=[20, 32], help='Index')
    parser.add_argument('--dst_temperature', default=1, type=float, help='temperature')
    parser.add_argument('--dst_loss_weight', default=1, type=float, help='weight of distillation')
    parser.add_argument('--full_dir', type=str, help='Index')
    parser.add_argument('--log_dir', type=str, help='Index')
    parser.add_argument('--slim_dir', type=str, help='Index')
    parser.add_argument('--warmup_dir', type=str, help='Index')
    parser.add_argument('--search_dir', type=str, help='Index')
    parser.add_argument('--teacher_dir', type=str, help='Index')
    args = parser.parse_args()


    Dataset = Cifar10 if args.dataset == 'cifar10' else Cifar100
    dataset = Dataset(args.data_path)
    n_class = dataset.n_class

    device = 'cuda:0'
    teacher = None
    if args.dst_flag:
        teacher_net = ResNet(args.teacher_net_index, n_class)
        teacher = Distiller(dataset, teacher_net, device, args, model_path=args.teacher_dir+'/6884.pth')

    if not args.prune_flag:
        net = ResNet(args.net_index, n_class)
        learner = FullLearner(dataset, net, device, args, teacher=teacher)
        # learner.train(n_epoch=args.num_epoch, save_path=args.full_dir)
        learner.load_model(os.path.join(args.full_dir, 'model_170.pth'))
        learner.test()
    else:
        net = ResNetGated(args.net_index, n_class)
        learner = DcpsLearner(dataset, net, device, args, teacher=teacher)

if __name__ == '__main__':
    main()
