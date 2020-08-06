import torch
import torch.nn as nn

class Dnas(nn.Module):
    def __init__(self, tau, searching, n_param, split_type):
        super(Dnas, self).__init__()
        if split_type not in ['fix_seg_size', 'fix_gp_number']:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')

        self.seg_sz = n_param if split_type == 'fix_seg_size' else None
        self.n_seg = n_param if split_type == 'fix_gp_number' else None

