#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 19:48:34 2020

@author: zhaoyu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from dataset.mnist import Mnist
from TTLayer import TTLayer

INIT_LR = 1e-2
MOMENTUM = 0.9
L2_REG = 0

class Lenet(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_classes):
        super(Lenet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.n_classes = n_classes
        self.fc_layers, self.bn_layers = self._build_hidden_layers()
        self.output_layer = self._build_output_layer()
        
    def _build_hidden_layers(self):
        # fc_layers = [TTLayer([8, 4, 8, 4], [7, 4, 7, 4], [2, 2, 2])]
        # bn_layers = [nn.BatchNorm1d(1024)]
        fc_layers = [nn.Linear(self.input_size, self.hidden_sizes[0], bias=False)]
        bn_layers = [nn.BatchNorm1d(self.hidden_sizes[0])]
        for i, _ in enumerate(self.hidden_sizes[:-1]):
            fc_layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1], bias=False))
            bn_layers.append(nn.BatchNorm1d(self.hidden_sizes[i+1]))
        return nn.ModuleList(fc_layers), nn.ModuleList(bn_layers)
    
    def _build_output_layer(self):
        return nn.Linear(self.hidden_sizes[-1], self.n_classes, bias=True)
        
    def forward(self, x):
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            x = F.relu(bn(fc(x)))
        return self.output_layer(x)

def metrics(criterion, outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    loss = criterion(outputs, labels)
    accuracy = correct / labels.size(0)
    return accuracy, loss

if __name__ == '__main__':
    n_feats = 784
    n_classes = 10
    
    dataset = Mnist('../dataset/data')
    train_batch_size = 256
    test_batch_size = 100
    train_loader = dataset.build_dataloader(train_batch_size, is_train=True)
    test_loader = dataset.build_dataloader(test_batch_size, is_train=False)

    device = torch.device('cpu')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hidden_sizes = [1024]
    lenet = Lenet(n_feats, hidden_sizes, n_classes).to(device)
    print(lenet)

    optimizer = optim.SGD(lenet.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)
    lr = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
    
    criterion = nn.CrossEntropyLoss()
    
    n_epoch = 20
    for epoch in range(n_epoch):
        lr.step()
        for i, data in enumerate(train_loader):
            batch_size = data[0].shape[0]
            images, labels = data[0].view(batch_size, -1).to(device), data[1].to(device)
            optimizer.zero_grad()
            
            outputs = lenet(images)
            accuracy, loss = metrics(criterion, outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(i+1, ' acc={0:.2f}, loss={1:.3f}'.format(accuracy*100, loss))
        print(epoch+1, 'finished')

        total_accuracy_sum = 0
        total_loss_sum = 0
        for i, data in enumerate(test_loader, 0):
            images, labels = data[0].view(test_batch_size, -1).to(device), data[1].to(device)
            outputs = lenet(images)
            accuracy, loss = metrics(criterion, outputs, labels)
            total_accuracy_sum += accuracy
            total_loss_sum += loss.item()
        avg_loss = total_loss_sum / len(test_loader)
        avg_acc = total_accuracy_sum / len(test_loader)
        print('acc= {0:.2f}, loss={1:.3f}'.format(avg_acc * 100, avg_loss))


    print('Finished Training')

