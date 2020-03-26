# -*- coding: utf-8 -*-

'''
1. GRU
2. CNN
3. GRU+CNN
4. Transformer
5. Star-Transformer
6. BT-Transformer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNN(nn.Module):
    '''
    hyperparameters:
        batch_size=32, init_lr=2.5e-2, weight_decay=1e-5, lr_decay=0.1: 83.50%
    '''
    def __init__(self, vocab_size, nemb, nhid, nclass, nlayer=1, dropout=0.2):
        super(RNN, self).__init__()
        self.nemb = nemb
        self.nhid = nhid
        self.nclass = nclass
        self.nlayer = nlayer
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=nemb)
        self.dropout1 = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_size=nemb, hidden_size=nhid, num_layers=nlayer, 
                dropout=0 if nlayer==1 else dropout, bidirectional=True)
        self.num_dir = 2
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(nhid*self.num_dir, nclass)
        
    def forward(self, x, hidden_state):
        emb = self.emb(x)
        emb = self.dropout1(emb)
        rnn_output, hidden_state = self.rnn(emb, hidden_state)
        output = self.dropout2(rnn_output[-1,...])
        logits = self.fc(output)
        return logits
    
    def init_hiddens(self, batch_size):
        return torch.zeros([self.nlayer*self.num_dir, batch_size, self.nhid])
    
   
        
