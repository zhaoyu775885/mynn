import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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

class DcpConfig():
    def __init__(self, n_param=1, split_type=TYPE_A, reuse_gate=None):
        self.n_param = n_param
        self.split_type = split_type
        self.reuse_gate = reuse_gate

    def copy(self, reuse_gate=None):
        dcfg = copy.copy(self)
        dcfg.reuse_gate = reuse_gate
        return dcfg

    def __str__(self):
        return '{0}; {1}; {2}'.format(self.n_param, self.split_type, self.reuse_gate)

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias, dcfg=None):
        # todo: share mask is not enough!
        # todo: consider wrap the additional parameters
        # todo: degenerate to common Conv2d while dcfg is None or abnormal
        # todo: actually, dcfg == None is not allowed
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

        self.dcfg = dcfg
        if dcfg is None:
            return

        if dcfg.split_type not in ['fix_seg_size', 'fix_gp_number', TYPE_A, TYPE_B]:
            raise ValueError('Only \'fix_seg_size\' and \'fix_gp_number\' are supported.')

        if dcfg.split_type == 'fix_seg_size' or dcfg.split_type == TYPE_B:
            self.seg_sz = dcfg.n_param
            self.n_seg = int(np.ceil(out_planes/dcfg.n_param))
        elif dcfg.split_type == 'fix_gp_number' or dcfg.split_type == TYPE_A:
            self.n_seg = dcfg.n_param
            self.seg_sz = int(np.ceil(out_planes/dcfg.n_param))

        self.mask = self.__init_mask(out_planes)
        self.gate = self.__init_gate(dcfg.reuse_gate)

    def __init_mask(self, out_planes):
        seg_sz_num = out_planes + self.n_seg - self.n_seg*self.seg_sz
        seg_sub_sz_num = self.n_seg - seg_sz_num
        seg_list = [self.seg_sz]*seg_sz_num + [self.seg_sz-1]*seg_sub_sz_num
        seg_tail_list = [sum(seg_list[:i+1]) for i in range(self.n_seg)]
        mask = torch.zeros(out_planes, self.n_seg)
        for col in range(self.n_seg):
            mask[:seg_tail_list[col], col] = 1
        # todo: determine whether to register mask
        return nn.Parameter(mask, requires_grad=False)

    def __init_gate(self, reuse_gate=None):
        gate = torch.zeros([self.n_seg]) if reuse_gate is None else reuse_gate
        return nn.Parameter(gate)

    def __gumbel_softmax(self, tau=1, searching=False):
        if searching:
            uniform_noise = torch.rand(self.n_seg)
            gumbel_noise = -torch.log(-torch.log(uniform_noise))
            return F.softmax((self.gate+gumbel_noise)/tau, dim=0)
        return F.softmax(self.gate/tau, dim=0)

    def forward(self, x, tau=1, searching=False, reuse_prob=None):
        y = self.conv(x)
        prob = self.__gumbel_softmax(tau, searching) if reuse_prob is None else reuse_prob
        pmask = torch.sum(self.mask * prob, dim=1)
        return y*pmask.view(1, len(pmask), 1, 1), prob

if __name__ == '__main__':
    gate = torch.tensor([0, 0, 0, 0], dtype=torch.double)
    dcfg = DcpConfig(n_param=4, split_type=TYPE_A, reuse_gate=gate)
    conv = Conv2d(3, 4, kernel_size=3, stride=1, padding=0, bias=False, dcfg=dcfg)
    print(conv.mask)
    print(conv.gate)

    x = torch.ones([1, 3, 4, 4])
    y = conv(x)
    print(y)

    for k, v in conv.named_parameters():
        print(k)


