# from random import shuffle
# import re
from select import select
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
from utility import n_blocks, device
import math
import random

maxPsnr = 0

n_epochs = 60
image_size = 32
scale_factor = 2
batch_size = 16
learning_rate = 0.0001
generations = 40
populationSize = 16
mutationProb = 0.2
elitismNumber = 8

nConv = [4, 6, 8]
nChannel = [16, 24, 32, 48, 64]
nRecursion = {1, 2, 3, 4}

pwd = os.getcwd()
train_hr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR")
train_lr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2")
test_hr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR")
test_lr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2")
check_hr_path = os.path.join(pwd, "Data", "DIV2K", "check", "HR")
check_lr_path = os.path.join(pwd, "Data", "DIV2K", "check", "LR")

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

class GEA:
    def __init__(self, G, p_size, mut_prob, T):
        self.mutationProb = mut_prob
        self.populationSize = p_size
        self.generations = G
        self.H = generatePopulation(p_size, n_blocks)
        self.population = [{'genome' : self.H[i], 'fitness' : 0} for i in range(p_size)]
        self.Loss = nn.L1Loss()
        self.PSNR = psnr.PSNR()
        self.total_train_sample = 0
        self.total_test_sample = 0
        self.train_loader = []
        self.test_loader = []

    def loadData(self):
        train_data = DIV2Kdataset(check_lr_path, check_hr_path, train_transform, test_transform)
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        self.total_train_sample = len(train_data)

        test_data = DIV2Kdataset(check_lr_path, check_hr_path, train_transform, test_transform)
        self.test_loader = DataLoader(
            dataset=test_data,
        )
        self.total_test_sample = len(test_data)
        # t = transforms.ToTensor()
        # print(len(test_data), len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]))
        # print(len(self.train_loader), len(self.test_loader))

    def train(self, _model):
        eta = 0.0625

        self.optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
        n_iteration = math.ceil(self.total_train_sample/batch_size)
        for epoch in range(n_epochs):
            for i, (lr, hr) in enumerate(self.train_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                y = _model(lr)

                loss = eta*self.Loss(y, hr)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (i+1)%1 == 0 and (epoch+1)%10 == 0:
                    print(f'epoch [{epoch+1}/{n_epochs}], step [{i+1}/{n_iteration}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

            if (epoch+1)%10 == 0:
                eta *= 2

    def test(self, _model):
        avgPsnr = 0
        avgLoss = 0
        total = 0

        with torch.no_grad():
            for i, (lr, hr) in enumerate(self.test_loader):
                lr = lr.to(device)
                hr = hr.to(device)
                # print(lr.shape)
                y = _model(lr)

                loss = self.Loss(y, hr)
                
                total += 1
                avgPsnr += self.PSNR(y, hr)
                avgLoss += loss
                if (i+1)%10 == 0:
                    print(f'step [{i+1}/{self.total_test_sample}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

            avgLoss /= total
            avgPsnr /= total

            # test = [x for x in self.test_loader]

            # tra = transforms.ToPILImage()
            # eg = _model(test[0][0].to(device))
            # im_lr = tra(test[0][0][0])
            # im_hr = tra(test[0][1][0])
            # img = tra(eg[0])
            # im_lr.show()
            # im_hr.show()
            # img.show()

            return avgPsnr, avgLoss

    def evaluate(self, index):
        global maxPsnr
        print(self.population[index]['genome'])
        _model = GenerateModel(self.population[index]['genome'], image_size, scale_factor).to(device)
        self.train(_model)
        psnr, loss = self.test(_model)
        if(maxPsnr < psnr):
            PATH = ''
            PATH = PATH.join(self.population[index]['genome'])
            PATH = str(round(psnr.item(), 3)) + '-' + PATH + '.pth'
            PATH = os.path.join('parameters', PATH)
            maxPsnr = psnr
            torch.save(_model.state_dict(), PATH)
        self.population[index]['fitness'] = psnr
        print(f'fitness/psnr value of {index+1}th population : {psnr}')

    def evolution(self):
        print('Loading Data...')
        self.loadData()
        print('Data Loaded')
        for generation in range(self.generations):
            print('generation', generation+1)
            for i in range(self.populationSize):
                self.evaluate(i)
            sorted(self.population, key = lambda p: p['fitness'], reverse=True)
            tempPopulation = [{'genome':[], 'fitness':0} for i in range(self.populationSize)]
            for i in range(self.populationSize//2):
                randInt = random.randint(0, 100)
                if(100*self.mutationProb >= randInt):
                    randInt = random.randint(0, n_blocks-1)
                    type = self.population[i]['genome'][randInt]
                    blockType = ['s', 'g', 'c']
                    blockType.remove(type)
                    tempPopulation[i]['genome'] = self.population[i]['genome']
                    tempPopulation[i]['genome'][randInt] = random.choice(blockType)
            upto = self.populationSize//2
            for i in range(1, (self.populationSize+1)//2):
                if(upto >= self.populationSize):
                    break
                for j in range(i):
                    if(upto >= self.populationSize):
                        break
                    crossOver = self.population[i]['genome'][:n_blocks//2] + self.population[j]['genome'][n_blocks//2:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
                    if(upto >= self.populationSize):
                        break
                    crossOver = self.population[j]['genome'][:n_blocks//2] + self.population[i]['genome'][n_blocks//2:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
            if(upto != self.populationSize):
                print(f'expected upto be {self.populationSize} but is {upto}')
            tempPopulation = self.population

find = GEA(generations, populationSize, mutationProb, elitismNumber)
find.evolution()