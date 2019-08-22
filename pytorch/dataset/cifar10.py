import torch
import torchvision
import torchvision.transforms as transforms

class Cifar10():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def build_dataloader(self, batch_size, is_train=True):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # explain the meaning
        dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=is_train, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=True, num_workers=4)
        return dataloader