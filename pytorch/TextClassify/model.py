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

class RNN(nn.Module):
    def __init__(self, vocab_size, nemb, nhid, nclass, nlayer=2):
        super(RNN, self).__init__()
        self.nemb = nemb
        self.nhid = nhid
        self.nclass = nclass
        self.nlayer = nlayer
        self.emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=nemb)
        self.rnn = nn.GRU(input_size=nemb, hidden_size=nhid, num_layers=nlayer)
        self.fc = nn.Linear(nhid, nclass)
        
    def forward(self, x, hidden_state):
        emb = self.emb(x)
        rnn_output, hidden_state = self.rnn(emb, hidden_state)
        logits = self.fc(rnn_output)
        return logits[-1,...]
    
    def init_hiddens(self, batch_size):
        return torch.zeros([self.nlayer, batch_size, self.nhid])
        
