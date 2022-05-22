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
from utility import n_min_blocks, device, n_max_blocks
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
rdbPsnr = 0

nConv = [4, 6, 8]
nChannel = [16, 24, 32, 48, 64]
nRecursion = {1, 2, 3, 4}

blocks = ['s', 'g', 'c']

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
        self.H = generatePopulation(p_size, n_min_blocks-1)
        self.population = [{'genome' : self.H[i], 'fitness' : 0} for i in range(p_size)]
        self.Loss = nn.L1Loss()
        self.PSNR = psnr.PSNR()
        self.total_train_sample = 0
        self.total_test_sample = 0
        self.train_loader = []
        self.test_loader = []
        self.BlockCreditMatrix = {'s' : [0 for _ in range(20)],
                                  'g' : [0 for _ in range(20)],
                                  'c' : [0 for _ in range(20)]}

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

                # if (i+1)%1 == 0 and (epoch+1)%10 == 0:
                    # print(f'epoch [{epoch+1}/{n_epochs}], step [{i+1}/{n_iteration}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

            if (epoch+1)%10 == 0:
                eta *= 2

    def test(self, _model):
        avgPsnr = 0
        avgLoss = 0
        total = 0

        with torch.no_grad():
            for i, (lr, hr) in enumerate(self.test_loader):
                # print('.', end='')
                lr = lr.to(device)
                hr = hr.to(device)
                # print(lr.shape)
                y = _model(lr)

                loss = self.Loss(y, hr)
                
                total += 1
                avgPsnr += self.PSNR(y, hr)
                avgLoss += loss
                # if (i+1)%10 == 0:
                #     print(f'step [{i+1}/{self.total_test_sample}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

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

    def evaluate(self, genome, index):
        global maxPsnr

        credits = [0 for _ in range(len(genome)+1)]
        credits[0] = rdbPsnr
        for i in range(1, len(genome)+1):
            _model = GenerateModel(genome[:i], image_size, scale_factor).to(device)
            self.train(_model)
            psnr, loss = self.test(_model)
            if(maxPsnr < psnr):
                PATH = ''
                PATH = PATH.join(genome)
                PATH = str(round(psnr.item(), 3)) + '-' + PATH + '.pth'
                PATH = os.path.join('parameters', PATH)
                maxPsnr = psnr
                torch.save(_model.state_dict(), PATH)
            credits[i] = psnr
        self.population[index]['fitness'] = psnr
        print(f'Population : {index+1}, Psnr : {psnr}')
        credits = [credits[i] - credits[i-1] for i in range(len(credits)-1, 0, -1)]
        return credits[1:]

    def evolution(self):
        global rdbPsnr

        alpha = 0.9
        epsilon = 0.001
        print('Loading Data...')
        self.loadData()
        print('Data Loaded')
        # blockLen = 
        # prevPsnr = {'s' : [0 for _ in range(20)],
        #             'g' : [0 for _ in range(20)],
        #             'c' : [0 for _ in range(20)]}
        _model = GenerateModel(['r'], image_size, scale_factor).to(device)
        self.train(_model)
        rdbPsnr, loss = self.test(_model)
        # print(rdbPsnr)
        

        for generation in range(self.generations):
            # for p in self.population:
                # print(p)
            print('Generation :', generation+1)
            # for p in self.population:
            #     print(p)
            # break
            # k = 0
            for i, p in enumerate(self.population):
                # genome = p['genome']
                if len(p['genome']) < n_max_blocks:
                    p['genome'].append(random.choice(['s', 'g', 'c']))
                # print(p)
                # print(p['genome'], genome)
                credits = self.evaluate(p['genome'], i)
                # k += 1
                for i in range(len(credits)):
                    self.BlockCreditMatrix[p['genome'][i]][i] = alpha*self.BlockCreditMatrix[p['genome'][i]][i] + (1-alpha)*credits[i]

            sorted(self.population, key = lambda p : p['fitness'], reverse = True)
            Elites = self.population[:self.populationSize//2]
            tempPopulation = self.population[:]
            for i in range(self.populationSize//2):
                # mutate
                totalCredit = sum([pow(self.BlockCreditMatrix[block][j], 2) for j, block in enumerate(Elites[i]['genome'])])
                # print([self.BlockCreditMatrix[block][j] for j, block in enumerate(Elites[i]['genome'])])
                for j, block in enumerate(Elites[i]['genome']):
                    selectProb = pow(self.BlockCreditMatrix[block][j], 2)/totalCredit
                    if random.random() < selectProb and random.random() < mutationProb:
                        print('ho rha hai')
                        blockType = ['s', 'g', 'c']
                        blockType.remove(block)
                        Elites[i]['genome'][j] = random.choice(blockType)
                tempPopulation[i] = Elites[i]
                # crossover

            upto = self.populationSize//2
            for i in range(1, (self.populationSize+1)//2):
                if(upto >= self.populationSize):
                    break
                for j in range(i):
                    if(upto >= self.populationSize):
                        break
                    crossOver = self.population[i]['genome'][:len(self.population[i]['genome'])//2] + self.population[j]['genome'][len(self.population[i]['genome'])//2:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
                    if(upto >= self.populationSize):
                        break
                    crossOver = self.population[j]['genome'][:len(self.population[i]['genome'])//2] + self.population[i]['genome'][len(self.population[i]['genome'])//2:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
            if(upto != self.populationSize):
                print(f'expected upto be {self.populationSize} but is {upto}')
            self.population = tempPopulation[:]

            # print(self.BlockCreditMatrix)
            # for p in self.population:
            #     print(p)
            # return Elites
            # print()
            # prevPsnr = [0]
            # last = 0

        return Elites

find = GEA(generations, populationSize, mutationProb, elitismNumber)
for e in find.evolution():
    print(e)