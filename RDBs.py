from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvRelu(nn.Module):
    def __init__(self, Cin, Cout, kSize, group=1):
        super(ConvRelu, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, Cout, kernel_size=kSize, padding=(kSize//2), stride=1, groups=group),
            # nn.ReLU(inplace=False)
            # nn.ReLU()

        ])

    def forward(self, x):
        return F.relu6(self.conv(x))

class basicRDB(nn.Module):
    def __init__(self, Cin, Cout, nConv):
        super(basicRDB, self).__init__()
        self.nConv = nConv
        self.conv1 = nn.Conv2d(Cout, Cout, kernel_size=1, padding=0, stride=1)
        self.convRelu = ConvRelu(Cin, Cout, 3)
    
    def forward(self, x):
        y = x
        for _ in range(self.nConv):
            y = self.convRelu(y) + y
        return self.conv1(y) + x

class SRDB(nn.Module):
    def __init__(self, Cin, Cout, nConv):
        super(SRDB, self).__init__()
        self.nConv = nConv
        self.convSRDB = nn.Sequential(*[
            nn.Conv2d(Cout, Cout, 1, padding=0, stride=1),
            ConvRelu(Cout, Cout, 3)
        ])
        self.conv1 = nn.Conv2d(Cout, Cout, 1, padding=0, stride=1)
        self.conv2 = nn.Conv2d(Cin, Cout, 1, padding=0, stride=1)
        self.convRelu = ConvRelu(Cin, Cout, 3)
    
    def forward(self, x):
        y = self.convRelu(x)
        x = self.conv2(x)
        # print(len(x[0]), len(y[0]))
        y += x
        for _ in range(self.nConv - 1):
            y = self.convSRDB(y) + y
        return self.conv1(y) + x

class CRDB(nn.Module):
    def __init__(self, Cin, Cout, nConv, nRec, OutputSize):
        super(CRDB, self).__init__()
        self.nRec = nRec
        self.nConv = nConv
        self.pool = nn.MaxPool2d(kernel_size=2)
        # Cin //= 2
        # Cout //= 2
        # print(Cin, Cout)
        self.convCRDB = nn.Sequential(*[
            nn.Conv2d(Cout, Cout, 1, padding=0, stride=1),
            ConvRelu(Cout, Cout, 3)
        ])
        self.UpSample = nn.Upsample(size=(OutputSize, OutputSize))
        self.conv1 = nn.Conv2d(Cout, Cout, 1, padding=0, stride=1)
        self.convRelu = ConvRelu(Cout, Cout, 3)
        # self.out = OutputSize

    def recursive(self, x, layer, n_rec):
        for _ in range(n_rec):
            x = layer(x)
        return x
    
    def forward(self, x):
        y = self.pool(x)
        y = self.recursive(y, self.convRelu, self.nRec) + y
        for _ in range(self.nConv - 1):
            y = self.recursive(y, self.convCRDB, self.nRec) + y
        y = self.conv1(self.UpSample(y))
        # print(y.shape, x.shape, self.out)
        return y + x

class GRDB(nn.Module):
    def __init__(self, Cin, Cout, nConv, nGroup):
        super(GRDB, self).__init__()
        self.nConv = nConv
        self.nGroup = nGroup
        self.conv1 = nn.Conv2d(Cout, Cout, 1, padding=0, stride=1)
        self.suffle = nn.ChannelShuffle(nGroup)
        self.convRelu = ConvRelu(Cin, Cout, kSize=3, group=nGroup)
        # self.groupConv = nn.conv
    
    def forward(self, x):
        y = self.convRelu(x) + x
        for _ in range(self.nConv - 1):
            # y = self.suffle(self.convRelu(y)) + y
            y = self.convRelu(y) + y
        return self.conv1(y) + x

class UpsampleBlock(nn.Module):
    def __init__(self, Cin, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(Cin, Cin*scale_factor**2, kernel_size=3, padding=1, stride=1)
        self.ps = nn.PixelShuffle(scale_factor)
        # self.act = nn.ReLU(inplace=False)
        # self.act = nn.ReLU()

    def forward(self, x):
        return F.relu6(self.ps(self.conv(x)))
