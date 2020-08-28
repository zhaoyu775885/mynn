import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet import _weights_init

import dcps.DNAS as dnas

cfg = {
    20: [3, 3, 3],
    32: [5, 5, 5],
    56: [9, 9, 9],
    110: [18, 18, 18],
    164: [27, 27, 27]
}

class ResidualBlockLite(nn.Module):
    '''
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    '''
    def __init__(self, in_planes, out_planes_list, stride=2, dcfg=None):
        super(ResidualBlockLite, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        assert dcfg is not None
        self.dcfg = dcfg
        self.dcfg_nonreuse = dcfg.copy()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = dnas.Conv2d(in_planes, out_planes_1, kernel_size=3, stride=stride, padding=1, bias=False,
                                 dcfg=self.dcfg_nonreuse)
        self.bn1 = nn.BatchNorm2d(out_planes_1)
        self.shortcut = None
        if stride != 1 or len(out_planes_list)>2 or in_planes != out_planes_2:
            self.shortcut = dnas.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride, padding=0,
                                        bias=False, dcfg=self.dcfg_nonreuse)
            self.dcfg.reuse_gate = self.shortcut.gate
        self.conv2 = dnas.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=1, padding=1, bias=False,
                                 dcfg=self.dcfg)

    def forward(self, x, tau=1, noise=False, reuse_prob=None):
        prob = reuse_prob
        shortcut = x
        x = F.relu(self.bn0(x))
        flops_list = []
        prob_list = []
        if self.shortcut is not None:
            shortcut, prob, shortcut_flops = self.shortcut(x, tau, noise, p_in=prob)
            prob_list.append(prob)
            flops_list.append(shortcut_flops)
        x, p0, conv1_flops = self.conv1(x, tau, noise, p_in=prob)
        prob_list.insert(0, p0)
        flops_list.insert(0, conv1_flops)
        x = F.relu(self.bn1(x))
        x, prob, conv2_flops = self.conv2(x, tau, noise, reuse_prob=prob, p_in=p0)
        prob_list.insert(1, prob)
        flops_list.insert(1, conv2_flops)
        x += shortcut
        # todo: the order of prob and flops should correspond to the order of channels
        return x, prob, prob_list, flops_list

class BottleneckLite(nn.Module):
    pass

class ResNetGated(nn.Module):
    def __init__(self, n_layer, n_class, channel_lists, dcfg):
        super(ResNetGated, self).__init__()
        self.channel_lists = channel_lists
        self.base_n_channel = channel_lists[0]
        self.n_class = n_class
        self.dcfg = dcfg
        self.cell_fn = ResidualBlockLite if n_layer < 50 else BottleneckLite
        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.conv0 = dnas.Conv2d(3, self.base_n_channel, 3, stride=1, padding=1, bias=False, dcfg=self.dcfg)
        self.dcfg.reuse_gate = self.conv0.gate
        self.block_n_cell = cfg[n_layer]
        self.block_list = self._block_layers()
        self.bn = nn.BatchNorm2d(channel_lists[-1][-1][-1])
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = dnas.Linear(channel_lists[-1][-1][-1], self.n_class, dcfg=self.dcfg)
        self.apply(_weights_init)

    def _block_fn(self, in_planes, out_planes_lists, n_cell, strides):
        blocks = [self.cell_fn(in_planes, out_planes_lists[0], strides, self.dcfg)]
        for i in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes_lists[i-1][-1], out_planes_lists[i], 1, self.dcfg))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i == 0:
                block = self._block_fn(self.base_n_channel, self.channel_lists[i+1], n_cell, 1)
                block_list.append(block)
            else:
                in_planes = self.channel_lists[i][-1][-1]
                block = self._block_fn(in_planes, self.channel_lists[i + 1], n_cell, 2)
                block_list.append(block)
        return nn.ModuleList(block_list)

    def forward(self, x, tau=1, noise=False):
        x, prob, flops = self.conv0(x, tau, noise)
        prob_list = [prob]
        flops_list = [flops]
        for i, blocks in enumerate(self.block_list):
            for block in blocks:
                x, prob, blk_prob_list, blk_flops_list = block(x, tau, noise, prob)
                flops_list += blk_flops_list
                prob_list += blk_prob_list
                prob = blk_prob_list[-1]
        x = F.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x, fc_flops = self.fc(x, p_in=prob)
        flops_list += [fc_flops]
        return x, prob_list, torch.sum(torch.stack(flops_list)), flops_list

def ResNet20Gated(n_classes):
    dcfg = dnas.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
    channel_list_20 = [16,
                         [[16, 16], [16, 16], [16, 16]],
                         [[32, 32, 32], [32, 32], [32, 32]],
                         [[64, 64, 64], [64, 64], [64, 64]]
                         ]
    return ResNetGated(20, n_classes, channel_list_20, dcfg)

def ResNet32Gated(n_classes):
    dcfg = dnas.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
    channel_list_32 = [16,
                       [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
                       [[32, 32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
                       [[64, 64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]
                       ]
    return ResNetGated(32, n_classes, channel_list_32, dcfg)

def ResNet56Gated(n_classes):
    dcfg = dnas.DcpConfig(n_param=8, split_type=dnas.TYPE_A, reuse_gate=None)
    channel_list_56 = [16,
                       [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16]],
                       [[32, 32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]],
                       [[64, 64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]
                       ]
    return ResNetGated(56, n_classes, channel_list_56, dcfg)