# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadStarAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_size=3):
        super(MultiheadStarAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = self.embed_dim // self.num_heads
        self.kernel_size = kernel_size
        assert self.kdim*num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.vdim = self.kdim
        self.q_proj = nn.Linear(self.embed_dim, self.kdim*self.num_heads)
        self.k_proj = nn.Linear(self.embed_dim, self.kdim*self.num_heads)
        self.v_proj = nn.Linear(self.embed_dim, self.vdim*self.num_heads)
        self.o_proj = nn.Linear(self.vdim*self.num_heads, self.embed_dim)
        self.unfold = nn.Unfold(kernel_size=[self.kernel_size, 1], padding=[self.kernel_size//2, 0])
        
    def forward(self, x, e, relay=None):
        '''
        input: x, e [ L, B, D ] s [1, B, D]
        '''
        leng, bsz, d_model = x.shape
        assert d_model==self.embed_dim, "d_model mush equal self.embed_dim"
        
        relay = x.mean(0, keepdim=True) if relay==None else relay
        s = relay.expand(leng, -1, -1)
        e, s = e.unsqueeze(0), s.unsqueeze(0) # e, s:[1,L,B,D]
        cores = torch.cat([e, s], dim=0) # cores: [2,L,B,D]
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        k_c, v_c = self.k_proj(cores), self.v_proj(cores)
        k_r, v_r = self.k_proj(relay), self.v_proj(relay)
        
        ring = self._ring_attn(q, k, v, k_c, v_c)
        star = self._star_attn(relay, torch.cat([k, k_r], 0), torch.cat([v, v_r], 0))
        return self.o_proj(ring), self.o_proj(star)
        
    
    def _star_attn(self, q, k, v):
        q, k, v = q.permute(1, 0, 2), k.permute(1, 2, 0), v.permute(1, 0, 2)
        logits = torch.matmul(q, k)/np.sqrt(self.kdim)
        alphas = F.softmax(logits, -1)
        att = torch.matmul(alphas, v).permute(1, 0, 2)
        return att
    
    def _ring_attn(self, q, k, v, k_c, v_c):
        # [L,B,D] -> [B,D,L]
        leng, bsz, d_model = q.shape
        q, k, v = q.permute(1,2,0), k.permute(1,2,0), v.permute(1,2,0)
        k_c, v_c = k_c.permute(2,3,0,1), v_c.permute(2,3,0,1)
        q = q.unsqueeze(2)
#        print(q.shape)
        # Due the Unfold requiring 4-D inputs, extend_dim [k, v] to 4-D [B, D, L, 1] 
        # and then unfold to [B, kernel_size*D, L] -> [B, D, kernel_size, L]
        k = self.unfold(k.unsqueeze(-1)).reshape([bsz, self.kdim, self.kernel_size, leng])
        v = self.unfold(v.unsqueeze(-1)).reshape([bsz, self.kdim, self.kernel_size, leng])
        # [2,L,B,D] -> [B,D,2,L]
        # [B, D, kernel_size+2, L]
        k, v = torch.cat([k, k_c], -2), torch.cat([v, v_c], -2)
#        print(q.shape, v.shape)
        alphas = F.softmax((q*k).sum(1, keepdim=True)/np.sqrt(self.kdim), 2)
#        print(alphas.shape)
        # [B, D, L] -> [L, B, D]
        att = (alphas * v).sum(2).permute(2, 0, 1)#.reshape(bsz, self.kdim, L, 1)
#        print(att.shape)
        return att

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
