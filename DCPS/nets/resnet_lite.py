import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.resnet import _weights_init

cfg = {
    20: [3, 3, 3],
    32: [5, 5, 5],
    56: [9, 9, 9],
    110: [18, 18, 18],
    164: [27, 27, 27]
}

channel_list_full = [16,
                     [[16, 16], [16, 16], [16, 16]],
                     [[32, 32, 32], [32, 32], [32, 32]],
                     [[64, 64, 64], [64, 64], [64, 64]]
                     ]

class ResidualBlockLite(nn.Module):
    '''
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    '''
    def __init__(self, in_planes, out_planes_list, stride=2):
        super(ResidualBlockLite, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes_1, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes_1)
        self.conv2 = nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes_2)
        self.shortcut = None
        if stride != 1 and len(out_planes_list)>2:
            self.shortcut = nn.Conv2d(in_planes, out_planes_list[-1], kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn0(x))
        if self.shortcut != None:
            shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x += shortcut
        return x

class BottleneckLite(nn.Module):
    pass

class ResNetLite(nn.Module):
    def __init__(self, n_layer, n_class, channel_lists):
        super(ResNetLite, self).__init__()
        self.channel_lists = channel_lists
        self.base_n_channel = channel_lists[0]
        self.n_class = n_class
        self.cell_fn = ResidualBlockLite if n_layer < 50 else BottleneckLite
        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.conv0 = nn.Conv2d(3, self.base_n_channel, 3, stride=1, padding=1, bias=False)
        self.block_n_cell = cfg[n_layer]
        self.block_list = self._block_layers()
        self.bn_1 = nn.BatchNorm2d(self.base_n_channel*(2**(len(self.block_n_cell)-1)))

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(self.base_n_channel*(2**(len(self.block_n_cell)-1)), self.n_class)
        self.apply(_weights_init)

    def _block_fn(self, in_planes, out_planes_lists, n_cell, strides):
        blocks = [self.cell_fn(in_planes, out_planes_lists[0], strides)]
        for i in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes_lists[i-1][-1], out_planes_lists[i], 1))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i==0:
                block_list.append(self._block_fn(self.base_n_channel, self.channel_lists[i+1], n_cell, 1))
            else:
                in_planes = self.channel_lists[i][-1][-1]
                block_list.append(self._block_fn(in_planes, self.channel_lists[i+1], n_cell, 2))
        return nn.ModuleList(block_list)

    def forward(self, x):
        x = self.conv0(x)
        for blocks in self.block_list:
            for block in blocks:
                x = block(x)
        x = F.relu(self.bn_1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet20Lite(n_classes):
    return ResNetLite(20, n_classes, channel_list_full)