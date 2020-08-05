import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    '''
    out_planes_list contains the corresponding number of convs.
    out_planes_list: [c1, c2, c_shortcut], c2 == c_shortcut
    if len(out_planes_list) == 2:
        direct shortcut
    elif len(out_planes_list) == 3:
        shortcut with conv
    '''
    def __init__(self, in_planes, out_planes_list, stride=2):
        super(ResidualBlock, self).__init__()
        out_planes_1 = out_planes_list[0]
        out_planes_2 = out_planes_list[1]
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes_1, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes_1)
        self.conv2 = nn.Conv2d(out_planes_1, out_planes_2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes_2)
        self.shortcut = None
        if stride != 1 or len(out_planes_list)>2:
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

class ResNet(nn.Module):
    def __init__(self):
        pass