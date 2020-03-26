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
    
class Transformer(nn.Module):
    def __init__(self, vocab, nemb, nhead, nhid, nlayer, nclass, dropout=0.1):
        super(Transformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.ntoken = len(vocab)
        self.ninp = nemb
        self.nhead = nhead
        self.nhid = nhid
        self.nlayer = nlayer
        self.nclass = nclass
        self.dropout = dropout
        
        #self.encoder = nn.Embedding.from_pretrained(vocab.vectors, freeze=False)
        self.embedding = nn.Embedding(self.ntoken, self.ninp)
        #self.dropout1 = nn.Dropout(self.dropout)
        self.pos_encoder = PositionalEncoding(nemb, dropout)
        
        self.src_mask = None
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayer)
        self.dropout2 = nn.Dropout(self.dropout)
        #self.pool = nn.AdaptiveMaxPool1d(1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Linear(self.ninp, self.nclass)
        
        #self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones([sz, sz], dtype=torch.int))
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.2
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, hidden_state, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embedding(src) #* math.sqrt(self.ninp)
        #src = self.dropout1(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.permute(1, 2, 0)
        output = self.pool(output).squeeze(-1)
        #output = self.dropout2(output)
        output = self.decoder(output)
        return output

    def init_hiddens(self, batch_size):
        return torch.zeros([1, batch_size, self.nhid])
