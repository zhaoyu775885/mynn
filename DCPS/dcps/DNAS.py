import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
class Dnas(nn.Module):
    def __init__(self, tau, searching, n_param, split_type):
        super(Dnas, self).__init__()
        if split_type not in ['fix_seg_size', 'fix_gp_number']:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')

        self.seg_sz = n_param if split_type == 'fix_seg_size' else None
        self.n_seg = n_param if split_type == 'fix_gp_number' else None
'''
TYPE_A = 1
TYPE_B = 2

# dict for conv and gate
convbase = {}

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias,
                 n_param, split_type=TYPE_A, reuse_gate=None):
        super(Conv2d, self).__init__()
        if split_type not in ['fix_seg_size', 'fix_gp_number', TYPE_A, TYPE_B]:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=bias)

        if split_type == 'fix_seg_size' or split_type == TYPE_B:
            self.seg_sz = n_param
            self.n_seg = int(np.ceil(out_planes/n_param))
        elif split_type == 'fix_gp_number' or split_type == TYPE_A:
            self.n_seg = n_param
            self.seg_sz = int(np.ceil(out_planes/n_param))

        self.mask = self.__init_mask(out_planes)
        self.gate = self.__init_gate(reuse_gate)

    def __init_mask(self, out_planes):
        seg_sz_num = out_planes + self.n_seg - self.n_seg*self.seg_sz
        seg_sub_sz_num = self.n_seg - seg_sz_num
        seg_list = [self.seg_sz]*seg_sz_num + [self.seg_sz-1]*seg_sub_sz_num
        seg_tail_list = [sum(seg_list[:i+1]) for i in range(self.n_seg)]
        mask = torch.zeros(out_planes, self.n_seg)
        for col in range(self.n_seg):
            mask[:seg_tail_list[col], col] = 1
        # todo: determine whether to register mask
        return mask
        # return nn.Parameter(mask, requires_grad=False)

    def __init_gate(self, reuse_gate):
        gate = torch.zeros([self.n_seg])
        if reuse_gate is not None and len(reuse_gate) == self.n_seg:
            gate = reuse_gate
        return nn.Parameter(gate)

    def gumbel_softmax(self, tau=1, searching=False):
        if searching:
            uniform_noise = torch.rand(self.n_seg)
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            return F.softmax((gumbel_noise+self.gate)/tau, dim=0)
        return F.softmax(self.gate/tau, dim=0)

    def forward(self, x, tau=1, searching=False):
        y = self.conv(x)
        prob = self.gumbel_softmax(tau, searching)
        pmask = torch.sum(self.mask * prob, dim=1)
        pmask = pmask.view(1, len(pmask), 1, 1)
        return y*pmask

if __name__ == '__main__':
    reuse_gate = torch.tensor([0, 0, 0, 1000], dtype=torch.double)
    conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False,
                  split_type=TYPE_B, n_param=4)
    print(conv.mask)
    print(conv.gate)
    x = torch.rand([1, 3, 32, 32])
    y = conv(x)
    # print(y)


