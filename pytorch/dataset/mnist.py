# -*- coding: utf-8 -*-

import torchvision
import torch
from torchvision import datasets, transforms

class Mnist():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def build_dataloader(self, batch_size, is_train=True):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        # explain the meaning
        dataset = torchvision.datasets.MNIST(root=self.data_dir, train=is_train, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return dataloader
    
if __name__ == '__main__':
    dataset = Mnist('./data')
    train_loader = dataset.build_dataloader(1, is_train=True)
    test_loader = dataset.build_dataloader(100, is_train=False)
    
    for i, data in enumerate(train_loader):
        print(i, len(data))