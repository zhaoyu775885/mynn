import torch
from abc import ABC
from abc import abstractmethod
from utils.writer import Writer
import os

BATCH_SIZE = 128

class AbstractLearner(ABC):
    def __init__(self, dataset, net, device, args):
        self.dataset = dataset
        self.net = net
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.args = args

        self.forward = self.net.to(self.device)
        self.loss_fn = self._setup_loss_fn()

        # logs save in text and visualize in tensorboard
        self.recoder = Writer(self.args.log_dir)

    def _build_dataloader(self, batch_size, is_train):
        return self.dataset.build_dataloader(batch_size, is_train)

    @abstractmethod
    def _setup_loss_fn(self):
        pass

    @abstractmethod
    def _setup_optimizer(self):
        pass

    @abstractmethod
    def train(self, **kwargs):
        pass

    @abstractmethod
    def test(self, **kwargs):
        pass

    def save_model(self, model_path):
        # todo: supplement the epoch info
        torch.save(self.net.state_dict(), os.path.join(model_path))

    def load_model(self, model_path):
        """
        make sure that the checkpoint on the disk contains all related variables
        in current network.
        """
        disk_state_dict = torch.load(os.path.join(model_path))
        try:
            self.net.load_state_dict(disk_state_dict)
        except RuntimeError:
            print('load model with conflicts, please check the difference.')
            state_dict = self.net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = disk_state_dict[key]
            self.net.load_state_dict(state_dict)
