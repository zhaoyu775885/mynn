import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    18: [2, 2, 2, 2],
    20: [3, 3, 3],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3]
}

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ResidualBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn0(x))
        if self.shortcut != None:
            shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x += shortcut
        return x

class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()

class ResNet(nn.Module):
    def __init__(self, n_layer, n_class, base_n_channel=16):
        super(ResNet, self).__init__()
        self.base_n_channel = base_n_channel
        self.n_class = n_class
        self.cell_fn = ResidualBlock if n_layer < 50 else Bottleneck
        if n_layer not in cfg.keys():
            print('Numer of layers Error: ', n_layer)
            exit(1)
        self.conv0 = nn.Conv2d(3, self.base_n_channel, 3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.base_n_channel)
        self.block_n_cell = cfg[n_layer]
        self.block_list = self._block_layers()
        self.bn_1 = nn.BatchNorm2d(self.base_n_channel*(2**(len(self.block_n_cell)-1)))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.base_n_channel*(2**(len(self.block_n_cell)-1)), self.n_class)        

    def _block_fn(self, in_planes, out_planes, n_cell, strides):
        blocks = [self.cell_fn(in_planes, out_planes, strides)]
        for _ in range(1, n_cell):
            blocks.append(self.cell_fn(out_planes, out_planes, 1))
        return nn.ModuleList(blocks)

    def _block_layers(self):
        block_list = []
        for i, n_cell in enumerate(self.block_n_cell):
            if i==0:
                block_list.append(self._block_fn(self.base_n_channel, self.base_n_channel*(2**i), n_cell, 1))
            else:
                block_list.append(self._block_fn(self.base_n_channel*(2**(i-1)), self.base_n_channel*(2**i), n_cell, 2))
        return nn.ModuleList(block_list)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        for blocks in self.block_list:
            for block in blocks:
                x = block(x)
        x = F.relu(self.bn_1(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet20():
    return ResNet(20, 10)

if __name__ == '__main__':
    net = ResNet(20, 10)
    x = torch.zeros([16, 3, 32, 32])
    y = net(x)
