# from random import shuffle
# import re
# from select import select
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

n_epochs = 100
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
#         'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2"),
#         'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2"),
#     },
#     'hr' : {
#         'check' : os.path.join(pwd, "Data", "DIV2K", "check", "HR"),
#         'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR"),
#         'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR"),
#     }
# }
# train_hr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR")
# train_lr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2")
# test_hr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR")
# test_lr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2")
# check_hr_path = os.path.join(pwd, "Data", "DIV2K", "check", "HR")
# check_lr_path = os.path.join(pwd, "Data", "DIV2K", "check", "LR")

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

class retrain:
    def __init__(self):
        self.Loss = nn.L1Loss()
        self.PSNR = psnr.PSNR()
        self.total_train_sample = 0
        self.total_test_sample = 0
        self.train_loader = []
        self.test_loader = []

    def loadData(self):
        print('Loading data...')
        train_data = DIV2Kdataset(paths['lr']['train'], paths['hr']['train'], train_transform, test_transform)
        self.train_loader = DataLoader(
            dataset=train_data,
            # batch_size=batch_size,
            # shuffle=True,
        )
        self.total_train_sample = len(train_data)

        test_data = DIV2Kdataset(paths['lr']['test'], paths['hr']['test'], train_transform, test_transform)
        self.test_loader = DataLoader(
            dataset=test_data,
        )
        self.total_test_sample = len(test_data)

        print('Data Loaded')

    def train(self, _model):
        global learning_rate

        optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        n_iteration = math.ceil(self.total_train_sample)
        for epoch in range(n_epochs):
            for i, (lr, hr) in enumerate(self.train_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                y = _model(lr)

                loss = self.Loss(y, hr)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1)%10 == 0 and (epoch+1)%10 == 0:
                    print(f'epoch [{epoch+1}/{n_epochs}], step [{i+1}/{n_iteration}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')
                
            if (epoch+1)%300 == 0:
                learning_rate /= 2
                optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

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
            im_lr.show()
            im_hr.show()
            img.show()

            return avgPsnr, avgLoss

def run(genome):
    print(f'Training model woth block sequence:')
    print(list(genome))
    find = retrain()
    # genome = ['g', 'g', 's', 's', 'g', 'c', 's', 's', 'c', 'g', 's', 'c']
    _model = GenerateModel(genome, image_size, scale_factor).to(device)
    path = psnr.isIn('parameters', genome)
    if path:
        path = os.path.join('parameters', path)
        _model.load_state_dict(torch.load(path))
    find.loadData()
    find.train(_model)
    psnr_, loss = find.test(_model)
    PATH = os.path.join('retrainedModel', f'{round(psnr_.item(), 3)}-{"".join(genome)}.pth')
    torch.save(_model.state_dict(), PATH)
    print('psnr ', round(psnr_.item(), 3), 'after training')

    # PATH = os.path.join('parameter', f'{round(psnr_, 3)}-{"".join(genome)}.pth')
    # torch.save(_model.state_dict(), PATH)