#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 22:18:38 2020

@author: zhaoyu
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import opt_einsum as oe

def test():
    a = torch.arange(60.).reshape(3,4,5)
    b = torch.arange(24.).reshape(4,3,2)
    c = oe.contract(a, [0,1,2], b, [1,0,3], [2,3])
    print(c)
    
class TTLayer(nn.Module):
    def __init__(self, row_dim_sizes, col_dim_sizes, ranks):
        super(TTLayer, self).__init__()
        self.row_dim_sizes = row_dim_sizes
        self.col_dim_sizes = col_dim_sizes
        assert(len(self.row_dim_sizes) == len(self.col_dim_sizes))
        self.n_dim = len(self.row_dim_sizes)
        self.shape = [np.prod(self.row_dim_sizes), np.prod(col_dim_sizes)]
        self.size = [np.prod(self.shape)]
        assert(self.n_dim == len(ranks)+1)
        self.ranks = [1]+ranks+[1]
        self.cores = self._build_random_cores()
        
    def _build_random_cores(self):
        cores = []
        for _ in range(self.n_dim):
            shape = [self.row_dim_sizes[_], self.col_dim_sizes[_], self.ranks[_], self.ranks[_+1]]
            cores.append(torch.rand(size=shape, requires_grad=True))
        return cores        
        
    def forward(self, x):
        if len(x) != self.shape[1]:
            print('error: shape mismatch, ttmatrix\'s column_size={0}, while vector_len={1} '
                  .format(self.shape[1], len(x)))
            return -1
        tensor_v = x.view(self.col_dim_sizes+[1])
        length = len(tensor_v.shape)
        sizeA = [i for i in range(length)]
        for d in range(self.n_dim-1, -1, -1):
            size_output = [i for i in range(d)] + [i for i in range(d+1,length-1)] + [length, length+1]
            tensor_v = oe.contract(tensor_v, sizeA, self.cores[d], [length, d, length+1, length-1], size_output)
        tensor_v = torch.squeeze(tensor_v, -1)
        y = tensor_v.permute(dims=[i for i in range(self.n_dim-1, -1, -1)]).reshape(-1)
        return y
        
if __name__ == '__main__':
    row_sizes = [2, 3]
    col_sizes = [2, 4]
    ranks = [2]    
    ttlayer = TTLayer(row_sizes, col_sizes, ranks)
    x = torch.ones([np.prod(col_sizes)])
    y = ttlayer(x)
        