# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MultiheadStarAttention(nn.Module):
    '''
    Args:
        embed_dim: total dimension of the model, $d_{model}$
        nhead: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
    ToDo:
        multihead,
        dropout,
        initlize
    '''
    def __init__(self, embed_dim, nhead, kdim=None, vdim=None, kernel_size=3, dropout=0.0):
        super(MultiheadStarAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        if kdim is None:
            assert nhead*(embed_dim//nhead)==embed_dim, "embed_dim must be divisible by nhead"
            self.kdim = embed_dim // nhead
        else:
            self.kdim = kdim
        if vdim is None:
            self.vdim = embed_dim // nhead
        else:
            self.kdim = kdim
        self.kernel_size = kernel_size
        assert self.kernel_size//2 == 1, "kernel_size must be odd"
        self.dropout = dropout
        self.nhead_kdim, self.nhead_vdim = nhead*self.kdim, nhead*self.vdim
        
        self.q_proj = nn.Linear(self.embed_dim, self.nhead_kdim)
        self.k_proj = nn.Linear(self.embed_dim, self.nhead_kdim)
        self.v_proj = nn.Linear(self.embed_dim, self.nhead_vdim)
        self.o_proj = nn.Linear(self.nhead_vdim, self.embed_dim)
        self.unfold = nn.Unfold(kernel_size=[self.kernel_size, 1], padding=[self.kernel_size//2, 0])
        
    def forward(self, x, e, relay=None):
        '''
        args: 
            x: hidden neurons [ L, B, D ] 
            e: embeddings [ L, B, D ] 
            relay: single node [1, B, D]
        '''
        assert x.shape == e.shape, "x and e must be with the same shape"
        L, B, D = x.shape
        Lr, Br, Dr = relay.shape
        assert Lr==1, "relay is only one node"
        assert D==self.embed_dim, "d_model mush equal self.embed_dim"
        
        relay = x.mean(0, keepdim=True) if relay==None else relay
        s = relay.expand(L, -1, -1)
        e, s = e.unsqueeze(0), s.unsqueeze(0) # e, s : [1,L,B,D]
        cores = torch.cat([e, s], dim=0)      # cores: [2,L,B,D]
        
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        k_c, v_c = self.k_proj(cores), self.v_proj(cores)
        ring = self._ring_attn(q, k, v, k_c, v_c)
        
        k_r, v_r = self.k_proj(relay), self.v_proj(relay)
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
    def __init__(self, embed_dim, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(StarTransformerLayer, self).__init__()
        self.self_star_attn = MultiheadStarAttention(embed_dim, nhead=nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(embed_dim)
        self.dropout2 = nn.Dropout(embed_dim)
        self.activation = _get_activation_fn(activation)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequnce to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
        
class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            output = self.norm(output)

        return output    
        
if __name__ == '__main__':
    #               L, B, D
    x = torch.ones([3, 4, 5])
    e = torch.ones([3, 4, 5])
    msa = MultiheadStarAttention(5, 1)
    x, relay = msa(x, e)
