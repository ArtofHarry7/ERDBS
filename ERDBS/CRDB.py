from pickletools import optimize
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResUnit(nn.Module):
    def __init__(self):
        super(ResUnit, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class CRDBUnit(nn.Module):
    def __init__(self):
        super(CRDBUnit, self).__init__()
        self.resUnit = ResUnit()
        self.conv = nn.Conv1d(3, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.resUnit(self.conv(x))

class CRDB(nn.Module):
    def __init__(self, recursion, rec_layers):
        super(CRDB, self).__init__()
        self.recursion = recursion
        self.rec_layers = rec_layers
        self.pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.resUnit = ResUnit()
        self.crdbUnit = CRDBUnit()
    
    def forward(self, x):
        
        return final

# batch-normalization?
# talk about implementation