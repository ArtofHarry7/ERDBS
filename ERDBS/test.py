# from random import shuffle
# import re
# from select import select
from logging import raiseExceptions
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from dataLoader import DIV2Kdataset
import psnr
# import PIL
from model import GenerateModel, generateGenome, generatePopulation, getModels
from utility import device
import math
# import random

maxPsnr = 0

n_epochs = 1000
image_size = 64
scale_factor = 2
batch_size = 16
learning_rate = 0.0001
# generations = 40
# populationSize = 16
# mutationProb = 0.2
# elitismNumber = 8

nConv = [4, 6, 8]
nChannel = [16, 24, 32, 48, 64]
nRecursion = {1, 2, 3, 4}

# pwd = os.getcwd()
# paths = {
#     'lr' : {
#         'check' : os.path.join(pwd, "Data", "DIV2K", "check", "LR"),
#         'b100' : os.path.join(pwd, "Data", "benchmark", "B100", "LR_bicubic", "X2"),
#         'set5' : os.path.join(pwd, "Data", "benchmark", "Set5", "LR_bicubic", "X2"),
#         'set14' : os.path.join(pwd, "Data", "benchmark", "Set14", "LR_bicubic", "X2"),
#         'urban100' : os.path.join(pwd, "Data", "benchmark", "Urban100", "LR_bicubic", "X2"),
#         'div2k' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2"),
#     },
#     'hr' : {
#         'check' : os.path.join(pwd, "Data", "DIV2K", "check", "HR"),
#         'b100' : os.path.join(pwd, "Data", "benchmark", "B100", "HR"),
#         'set5' : os.path.join(pwd, "Data", "benchmark", "Set5", "HR"),
#         'set14' : os.path.join(pwd, "Data", "benchmark", "Set14", "HR"),
#         'urban100' : os.path.join(pwd, "Data", "benchmark", "Urban100", "HR"),
#         'div2k' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR"),
#     }
# }
# check_hr_path = os.path.join(pwd, "Data", "DIV2K", "check", "HR")
# check_lr_path = os.path.join(pwd, "Data", "DIV2K", "check", "LR")
# B100_hr_path = os.path.join(pwd, "Data", "benchmark", "B100", "HR")
# B100_lr_path = os.path.join(pwd, "Data", "benchmark", "B100", "LR_bicubic", "X2")
# Set5_hr_path = os.path.join(pwd, "Data", "benchmark", "Set5", "HR")
# Set5_lr_path = os.path.join(pwd, "Data", "benchmark", "Set5", "LR_bicubic", "X2")
# Set14_hr_path = os.path.join(pwd, "Data", "benchmark", "Set14", "HR")
# Set14_lr_path = os.path.join(pwd, "Data", "benchmark", "Set14", "LR_bicubic", "X2")
# Urban100_hr_path = os.path.join(pwd, "Data", "benchmark", "Urban100", "HR")
# Urban100_lr_path = os.path.join(pwd, "Data", "benchmark", "Urban100", "LR_bicubic", "X2")

train_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.CenterCrop(image_size),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(90),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(image_size*scale_factor),
    # transforms.Resize((128, 128)),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

class TEST:
    def __init__(self):
        self.Loss = nn.L1Loss()
        self.PSNR = psnr.PSNR()
        self.total_train_sample = 0
        self.total_test_sample = 0
        self.train_loader = []
        self.test_loader = []

    def loadData(self, test_lr_path_, test_hr_path_):
        print('Loading data...')
        test_data = DIV2Kdataset(test_lr_path_, test_hr_path_, train_transform, test_transform)
        self.test_loader = DataLoader(
            dataset=test_data,
        )
        self.total_test_sample = len(test_data)

        print('Data Loaded')

    def test(self, _model):
        avgPsnr = 0
        avgLoss = 0
        total = 0

        with torch.no_grad():
            for i, (lr, hr) in enumerate(self.test_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                y = _model(lr)

                loss = self.Loss(y, hr)
                
                total += 1
                avgPsnr += self.PSNR(y, hr)
                avgLoss += loss
                if (i+1)%10 == 0:
                    print(f'step [{i+1}/{self.total_test_sample}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

            avgLoss /= total
            avgPsnr /= total

            test = [x for x in self.test_loader]

            tra = transforms.ToPILImage()
            eg = _model(test[0][0].to(device))
            im_lr = tra(test[0][0][0])
            im_hr = tra(test[0][1][0])
            img = tra(eg[0])
            # im_lr.show()
            # im_hr.show()
            img.show()

            return avgPsnr, avgLoss

# print(paths['lr']['b100'], paths['hr']['b100'])

def run(test_data, genome):
    print(f'Testing {test_data} images on model woth block sequence:')
    print(list(genome))
    Test = TEST()
    # genome = ['g', 'g', 's', 's', 'g', 'c', 's', 's', 'c', 'g', 's', 'c']
    _model = GenerateModel(genome, image_size, scale_factor).to(device)
    path = psnr.isIn('retrainedModel', genome)
    if path:
        path = os.path.join('retrainedModel', path)
        _model.load_state_dict(torch.load(path))
        Test.loadData(paths['lr'][test_data], paths['hr'][test_data])
        psnr_, loss = Test.test(_model)
        print('psnr ', round(psnr_.item(), 3), ' for ', test_data)
    else:
        raiseExceptions('ye genome To trained hi nahi hai bhai')