# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import SST2 as Dataset
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

        print('begin training data')
        self.dataset = Dataset(self.data_path)
        #self.dataset_train = Dataset(self.train_path)
        #self.dataset_test = Dataset(self.test_path)
        print('finished data building')
        
        # build dataloader
        self.trainloader = self._build_dataloader(BATCH_SIZE, train=True)
        self.testloader = self._build_dataloader(1, train=False)

        vocab_size = self.dataset.vocab_size
        ninp, nhid, nclass = 300, 512, 2
        # for CNN, RNN and CRNN
#        self.net = network(vocab_size, ninp, nhid, nclass).to(self.device)
        # for Transformer and Star-Transformer
        nhead, nlayer = 4, 1
        self.net = network(self.dataset.vocab, ninp, nhead, nhid, nlayer, nclass).to(self.device)        

        # setup loss function and optimizer
        self.criterion = self._loss_fn()
        self.opt = self._setup_optimizer()
        self.lr_scheduler = self._setup_lr_scheduler()

    def _build_dataloader(self, batch_size, train=True):
        return self.dataset.build_iterator(batch_size, train)
    
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
        pred_correct = 0
        total_cnt = 0
        total_loss = 0
        for epoch in range(n_epoch):
            self.net.train()
            for i, batch in enumerate(self.trainloader):
                batch_size = batch.text.shape[-1]
                hiddens = self.net.init_hiddens(batch_size).to(self.device)
                #print(i, batch.label, self.dataset_train.raw(batch.text))
                texts, labels = batch.text.to(self.device), batch.label.to(self.device)
                #print(texts.shape, labels.shape)
                self.opt.zero_grad()
                logits = self.net(texts, hiddens)
                #print(logits.shape, labels.shape)
                accuracy, loss = self.metrics(logits, labels)
                loss.backward()
                self.opt.step()
                pred_correct += accuracy * batch_size
                total_cnt += batch_size
                total_loss += batch_size*loss
                if (i+1) % 100 == 0:
                    #print(batch.text.shape)
                    print(i+1, ' acc={0:.2f}, loss={1:.3f}'.format(accuracy*100, loss))
            self.test()
            print(epoch+1, 'finished')
            self.lr_scheduler.step()
        print('Finished Training')

    def test(self):
        pred_correct = 0
        total_cnt = 0
        total_loss = 0
        self.net.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.testloader):
                batch_size = batch.text.shape[-1]
                hiddens = self.net.init_hiddens(batch_size).to(self.device)
                texts, labels = batch.text.to(self.device), batch.label.to(self.device)
                logits = self.net(texts, hiddens)
                accuracy, loss = self.metrics(logits, labels)
                pred_correct += accuracy*batch_size
                total_cnt += batch_size
                total_loss += loss
                '''
                if (i+1) % 1000 == 0:
                    print('{0}: test acc={1:.2f}, pred_cor={2}, total={3}'.format(i+1,
                        pred_correct * 100 / total, pred_correct, total))
                '''
            print('test acc={0:.2f}, loss={1:.2f}'.format(pred_correct * 100 / total_cnt, total_loss/total_cnt))
