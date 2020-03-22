# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadStarAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = self.embed_dim // self.num_heads
        assert self.head_dim*num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.vdim = self.kdim
        self.q_proj = nn.Linear(self.embed_dim, self.kdim)
        self.k_proj = nn.Linear(self.embed_dim, self.kdim)
        self.v_proj = nn.Linear(self.embed_dim, self.vdim)
        
    def forward(self, x, cores=None):
        '''
        x.shape: L, B, D
        '''
        L, B, _ = x.shape
        x = x.permute()
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if cores == None:
            k_cores = self.k_proj(cores)
            v_cores = self.v_proj(cores)
        
        
#    def forward(self, x, ax=None):
#        # x: B, H, L, 1, ax : B, H, X, L append features
#        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
#        B, H, L, _ = x.shape
#
#        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)
#
#        if ax is not None:
#            aL = ax.shape[2]
#            ak = self.WK(ax).view(B, nhead, head_dim, aL, L)
#            av = self.WV(ax).view(B, nhead, head_dim, aL, L)
#        q = q.view(B, nhead, head_dim, 1, L)
#        k = F.unfold(k.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
#            .view(B, nhead, head_dim, unfold_size, L)
#        v = F.unfold(v.view(B, nhead * head_dim, L, 1), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
#            .view(B, nhead, head_dim, unfold_size, L)
#        if ax is not None:
#            k = torch.cat([k, ak], 3)
#            v = torch.cat([v, av], 3)
#
#        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / NP.sqrt(head_dim), 3))  # B N L 1 U
#        att = (alphas * v).sum(3).view(B, nhead * head_dim, L, 1)
#
#        ret = self.WO(att)
#
#        return ret        

class StarTransformerLayer(nn.Module):
    r'''
    Only support encoder part, ring-star decoder is going to be verified.
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(StarTransformerLayer, self).__init__()
        self.self_star_attn = MultiheadStarAttention()
        self.
        
        
        
if __name__ == '__main__':
    print(123)