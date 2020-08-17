import torch
from datasets.cifar import Cifar10, Cifar100
from nets.resnet import ResNet20
from learner.full import FullLearner

BATCH_SIZE = 128
INIT_LR = 1e-1
MOMENTUM = 0.9
L2_REG = 1e-4

class distiller():
        def __init__(self, Dataset, Net, device='cuda:0', model_path='./model/models.pth'):
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            self.dataset = Dataset
            self.net = Net

            # build dataloader
            self.trainloader = self._build_dataloader(BATCH_SIZE, is_train=True)
            self.testloader = self._build_dataloader(100, is_train=False)

            # define forward computational
            self.forward = self.net.to(self.device)

            self.load_model(model_path)
            self.net.eval()

        def _build_dataloader(self, batch_size, is_train):
            return self.dataset.build_dataloader(batch_size, is_train)

        def load_model(self, path='./models/models.pth'):
            disk_state_dict = torch.load(path)
            try:
                self.net.load_state_dict(disk_state_dict)
            except RuntimeError:
                print('Dismatched models, please check the network.')
                state_dict = self.net.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = disk_state_dict[key]
                self.net.load_state_dict(state_dict)

        def metrics(self, outputs, labels):
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            loss = self.criterion(outputs, labels)
            accuracy = correct / labels.size(0)
            return accuracy, loss

        def infer(self, images):
            return self.forward(images)

        def test(self):
            total_accuracy_sum = 0
            total_loss_sum = 0
            for i, data in enumerate(self.testloader, 0):
                images, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.infer(images)
                accuracy, loss = self.metrics(outputs, labels)
                total_accuracy_sum += accuracy
                total_loss_sum += loss.item()
            avg_loss = total_loss_sum / len(self.testloader)
            avg_acc = total_accuracy_sum / len(self.testloader)
            print('acc= {0:.2f}, loss={1:.3f}\n'.format(avg_acc * 100, avg_loss))

if __name__ == '__main__':
    cifar100 = '/home/zhaoyu/Datasets/cifar100'
    dataset = Cifar100(cifar100)