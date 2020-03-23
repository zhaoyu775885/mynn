# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Sentiment as Dataset
from model import RNN
import os

BATCH_SIZE = 128
INIT_LR = 1e-1
MOMENTUM = 0.9
L2_REG = 1e-5

class Learner():
    def __init__(self, dataset_path, network):
        # set device & build dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.data_path = dataset_path
        self.train_path = os.path.join(self.data_path, 'train.tsv')
        self.test_path = os.path.join(self.data_path, 'test.tsv')
        
        print('begin training data')
        self.dataset_train = Dataset(self.train_path)
#        print('begin test data')
#        self.dataset_test = Dataset(self.test_path)
        print('finished data building')
        
        # build dataloader
        self.trainloader = self._build_dataloader(BATCH_SIZE, train=True)
#        self.testloader = self._build_dataloader(1, train=False)

        vocab_size = self.dataset_train.vocab_size
        ninp, nhid, nclass = 300, 512, 5
        self.net = network(vocab_size, ninp, nhid, nclass).to(self.device)

        # setup loss function and optimizer
        self.criterion = self._loss_fn()
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    def _build_dataloader(self, batch_size, train=True):
        if train:
            return self.dataset_train.build_iterator(batch_size, train)
        return self.dataset_test.build_iterator(batch_size, train)
    
    def _loss_fn(self):
        return nn.CrossEntropyLoss()
    
    def _setup_optimizer(self):
        return optim.SGD(self.net.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)
    
    def _setup_lr_scheduler(self):
        return torch.optim.lr_scheduler.MultiStepLR(self.opt, milestones=[10, 15, 20], gamma=0.1)
    
    def metrics(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        loss = self.criterion(outputs, labels)
        accuracy = correct / labels.size(0)
        return accuracy, loss
    
    def train(self, n_epoch=25):
        print('Begin training:')
        for epoch in range(n_epoch):
            self.net.train()
            for i, batch in enumerate(self.trainloader):
                batch_size = batch.Phrase.shape[-1]
                hiddens = self.net.init_hiddens(batch_size).to(self.device)
                print(i, batch.Sentiment, self.dataset_train.raw(batch.Phrase))
                texts, labels = batch.Phrase.to(self.device), batch.Sentiment.to(self.device)
                self.opt.zero_grad()
                logits = self.net(texts, hiddens)
                accuracy, loss = self.metrics(logits, labels)
                loss.backward()
                self.opt.step()
                if (i+1) % 100 == 0:
                    print(batch.Phrase.shape)
                    print(i+1, ' acc={0:.2f}, loss={1:.3f}'.format(accuracy*100, loss))
            print(epoch+1, 'finished')
            self.lr_scheduler.step()
#            self.test()
#        print('Finished Training')
#
#    def test(self):
#        pred_correct = 0
#        self.dataset_test.init_batch_loader()
#        n_iters = self.dataset_test.n_iters()
#        print(n_iters, 'iterations')
#        for i, data in enumerate(self.testloader):
#           feats, labels, lengths = data
#           local_max_length = max(lengths)
#           inputs = feats.to(self.device)
#           outputs = self.net(inputs[:, :local_max_length, :])
#           accuracy, loss = self.metrics(outputs[:,-1,:], labels.to(self.device))
#           pred_correct += accuracy
#           # if (i+1) % 1000 == 0:
#           #     print(i+1, accuracy)
#           if i == n_iters - 1:
#               print('test acc={0:.2f}'.format(pred_correct * 100 / n_iters))
#               break
