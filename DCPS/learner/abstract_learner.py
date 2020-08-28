import torch
from abc import ABC
from abc import abstractmethod
from recoder.writer import Writer

BATCH_SIZE = 128


class AbstractLearner(ABC):
    def __init__(self, dataset, net, device='cuda:1', log_path='./log', teacher=None):
        self.dataset = dataset
        self.net = net
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.teacher = teacher

        self.forward = self.net.to(self.device)

        # define loss function
        self.criterion = self._loss_fn()

        # logs save in text and visualize in tensorboard
        self.recoder = Writer(log_path)

    def _build_dataloader(self, batch_size, is_train):
        return self.dataset.build_dataloader(batch_size, is_train)

    @abstractmethod
    def _loss_fn(self):
        pass

    @abstractmethod
    def metrics(self, logits, labels, kd=None):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    def save_model(self, path):
        # todo: supplement the epoch info
        torch.save(self.net.state_dict(), path)

    def load_model(self, path):
        """
        make sure that the checkpoint on the disk contains all related variables
        in current network.
        """
        disk_state_dict = torch.load(path)
        try:
            self.net.load_state_dict(disk_state_dict)
        except RuntimeError:
            print('Dismatched models, please check the network.')
            state_dict = self.net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = disk_state_dict[key]
            self.net.load_state_dict(state_dict)

