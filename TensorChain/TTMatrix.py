# -*- coding: utf-8 -*-

import numpy as np

class TTMatrix(object):
    def __init__(self, row_dim_sizes, col_dim_sizes, ranks):
        self.row_dim_sizes = row_dim_sizes
        self.col_dim_sizes = col_dim_sizes
        self.shape = [np.prod(self.row_dim_sizes), np.prod(col_dim_sizes)]
        self.size = [np.prod(self.shape)]
        assert(len(self.row_dim_sizes) == len(self.col_dim_sizes))
        self.n_dim = len(self.row_dim_sizes)
        assert(self.n_dim == len(ranks)+1)
        self.ranks = [1]+ranks+[1]
        # each core is a 4-order tensors
        self.cores = self._build_random_cores()
#        self.cores = self._build_one_cores()
        
    def __getitem__(self, index):
        return self.cores[index]
    
    def _build_random_cores(self):
        cores = []
        for _ in range(self.n_dim):
            shape = [self.row_dim_sizes[_], self.col_dim_sizes[_], self.ranks[_], self.ranks[_+1]]
            cores.append(np.random.random(size=shape))
        return cores
        
    def _build_one_cores(self):
        cores = []
        for _ in range(self.n_dim):
            shape = [self.row_dim_sizes[_], self.col_dim_sizes[_], self.ranks[_], self.ranks[_+1]]
            tmp_core = np.ones(shape)
            cores.append(tmp_core)
        return cores
    
    def convert_idx2list(self, row_flag, idx):
        bases = self.row_dim_sizes if row_flag else self.col_dim_sizes
        idx_list = []
        for dividor in bases[::-1]:
            idx_list.append(idx % dividor)
            idx //= dividor
        return idx_list[::-1]
        
    def element(self, row_idx, col_idx):
        val = np.mat([[1]])
        for d in range(self.n_dim):
            val = np.matmul(val, self.cores[d][row_idx[d], col_idx[d]])            
        return val[0, 0]
    
    def full_matrix(self):
        fmat = np.zeros(self.shape)
        row_idx, col_idx = [], []
        for i in range(self.shape[0]):
            row_idx.append(self.convert_idx2list(True, i))
        for j in range(self.shape[1]):
            col_idx.append(self.convert_idx2list(False, j))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                fmat[i, j] = self.element(row_idx[i], col_idx[j])
        return fmat
    
    def info(self):
        for i, core in enumerate(self.cores):
            print(i, ': ', core.shape)
        print('matrix shape: ', self.shape)
        
def tt_dot(ttm, v):
    ttm_shape = ttm.shape
    # consider batch size
    if v.shape[-1] != ttm_shape[1]:
        print('error: shape mismatch, ttmatrix\'s column_size={0}, while vector_len={1} '
              .format(ttm_shape[1], len(v)))
        return -1
    tensor_v = np.expand_dims(v.reshape([len(v)]+ttm.col_dim_sizes), -1)
    print(tensor_v.shape)
    for d in range(ttm.n_dim, 0, -1):
        tensor_v = np.tensordot(tensor_v, ttm[d-1], axes=[[d,-1], [1,3]])
    tensor_v = np.squeeze(tensor_v, -1)
    y = np.transpose(tensor_v, axes=[0]+[i for i in range(ttm.n_dim, 0, -1)]).reshape(v.shape[0], -1)
    return y

def tt_dot_einsum(ttm, v):
    ttm_shape = ttm.shape
    if v.shape[-1] != ttm_shape[1]:
        print('error: shape mismatch, ttmatrix\'s column_size={0}, while vector_len={1} '
              .format(ttm_shape[1], len(v)))
        return -1
    tensor_v = np.expand_dims(v.reshape([len(v)]+ ttm.col_dim_sizes), -1)
    length = len(tensor_v.shape)
    sizeA = [i for i in range(length)]
    print(sizeA)
    for d in range(ttm.n_dim, 0, -1):
        size_output = [i for i in range(d)] + [i for i in range(d+1,length-1)] + [length, length+1]
        print(sizeA, size_output)
        tensor_v = np.einsum(tensor_v, sizeA, ttm[d-1], [length, d, length+1, length-1], size_output)
        print(d, tensor_v.shape)
    tensor_v = np.squeeze(tensor_v, -1)
    y = np.transpose(tensor_v, axes=[0]+[i for i in range(ttm.n_dim, 0, -1)]).reshape(v.shape[0], -1)
    return y
    
if __name__ == '__main__':
#    n_dims = [2, 3, 4, 5]
#    for n_dim in n_dims:
#        row_sizes = np.random.randint(low=2, high=5, size=[n_dim]).tolist()
#        col_sizes = np.random.randint(low=2, high=5, size=[n_dim]).tolist()
#        ranks = np.random.randint(low=2, high=5, size=[n_dim-1]).tolist()
#        
#        tmat = TTMatrix(row_sizes, col_sizes, ranks)
#        tmat.info()
#        fmat = tmat.full_matrix()
#        v = np.random.random(size=[np.prod(col_sizes)])
#        yt = tt_dot_einsum(tmat, v)
#        yf = np.dot(fmat, v)
#        print(np.linalg.norm(yt-yf, ord=2))

    print('==============')
    row_sizes = [2, 4]
    col_sizes = [2, 4]
    ranks = [2]
    
    tmat = TTMatrix(row_sizes, col_sizes, ranks)
    fmat = tmat.full_matrix()
    # batch_size = 2
    v = np.ones([2, np.prod(col_sizes)])
#    ttmatrix.info()
    
    yt = tt_dot_einsum(tmat, v)
    print(yt)
    
    yf = np.dot(fmat, v.T).T
    print(yf)
