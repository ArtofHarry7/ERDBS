import imp
import torch
import torch.nn as nn
from RDBs import *
from random import choices
from utility import device

# imSize = 32

class GenerateModel(nn.Module):
    def __init__(self, genome, image_size, scale_factor):
        super(GenerateModel, self).__init__()
        self.conv = ConvRelu(3, 32, 3)
        self.conv1x1 = nn.Conv2d(32, 3, kernel_size=1, padding=0)
        self.up = UpsampleBlock(3, scale_factor)
        # print(genome)
        self.pattern = genome
        self.RDB = {
            'r' : basicRDB(32, 32, 4),
            's' : SRDB(32, 32, 4), 
            'c' : CRDB(32, 32, 4, 2, image_size), 
            'g' : GRDB(32, 32, 4, 4)
        }
        self.linears = nn.ModuleList([self.RDB[rdb] for rdb in self.pattern])

    def forward(self, x):
        x = self.conv(x)
        skip = x
        y = torch.zeros(x.shape).to(device)
        for l in self.linears:
            x = l(x)
            y += x
        
        return self.up(self.conv1x1(y))

def getModels(P):
    models = []
    for genome in P:
        models.append(GenerateModel(genome).to(device))
    return models

def generateGenome(length):
    return choices(['s', 'c', 'g'], k = length)

def generatePopulation(size, genome_length):
    return [generateGenome(genome_length) for _ in range(size)]
