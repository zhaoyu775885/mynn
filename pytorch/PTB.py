#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:44:22 2020

@author: zhaoyu
"""
import torch
from torchtext import data
import torchtext.datasets as datasets
import torch.nn as nn
import torch.optim as optim

INIT_LR = 1
L2_REG = 5e-5
MOMENTUM = 0.9

class PTB():
    def __init__(self, root_dir, batch_size=32, length=100):
        self.root_dir = root_dir
        self.field = data.Field(sequential=True, lower=False)
        all_datasets = datasets.PennTreebank.splits(text_field=self.field, root=self.root_dir)
        self.train, self.valid, self.test = all_datasets
        self.train_iter = data.BPTTIterator(dataset=self.train, batch_size=batch_size, bptt_len=length)
        self.field.build_vocab(self.train)

    def iter(self):
        return self.train_iter

    def vocab_len(self):
        return len(self.field.vocab)


class RNNNet(nn.Module):
    def __init__(self, n_embeddings, hidden_size, n_classes):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.embedding = nn.Embedding(num_embeddings=n_embeddings, embedding_dim=300)
        self.hidden = nn.LSTM(input_size=300, hidden_size=self.hidden_size, num_layers=1)
        self.output = nn.Linear(self.hidden_size, self.n_classes)

    def forward(self, x):
        embeded = self.embedding(x)
        lstm = self.hidden(embeded)
        lstm_out = lstm[0]
        logits = self.output(lstm_out)
        return logits


if __name__ == '__main__':
    root_dir = './data'
    ptb = PTB(root_dir)
    train_iter = ptb.iter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rnn = RNNNet(ptb.vocab_len(), 1024, ptb.vocab_len()).to(device)

    # 0: data iterator
    # 1: loss function
    # 2: learning_rate schedule
    # 3: metrics evaluation
    # 4: traing process or training log visualization
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.SGD(rnn.parameters(), lr=INIT_LR, momentum=MOMENTUM, weight_decay=L2_REG)
    lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.5)

    n_epoch = 25
    for _ in range(n_epoch):
        print('the {}-th epoch'.format(_ + 1))
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            batch_text, batch_target = batch.text.to(device), batch.target.to(device)
            outputs = rnn.forward(batch_text)
            loss_val = loss_func(outputs.view(-1, outputs.size(2)), batch_target.view(-1).long())
            if (i + 1) % 100 == 0:
                print(i + 1, ' loss= ', loss_val)
            loss_val.backward()
            optimizer.step()
        lr.step()