import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from dataLoader import DIV2Kdataset
import psnr
from model import GenerateModel, generatePopulation
from utility import *
import math
import random

maxPsnr = psnr.updateMaxPsnr()

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
paths = {
    'lr' : {
        'check' : os.path.join(pwd, "Data", "DIV2K", "check", "LR"),
        'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2"),
        'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2"),
    },
    'hr' : {
        'check' : os.path.join(pwd, "Data", "DIV2K", "check", "HR"),
        'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR"),
        'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR"),
    }
}
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
        train_data = DIV2Kdataset(paths['lr']['check'], paths['hr']['check'], train_transform, test_transform)
        self.train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        self.total_train_sample = len(train_data)

        test_data = DIV2Kdataset(paths['lr']['check'], paths['hr']['check'], train_transform, test_transform)
        self.test_loader = DataLoader(
            dataset=test_data,
        )
        self.total_test_sample = len(test_data)

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
                #     print(f'epoch [{epoch+1}/{n_epochs}], step [{i+1}/{n_iteration}], loss: {loss.item():.4f}, psnr: {self.PSNR(y, hr)}')

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


            return avgPsnr, avgLoss

    def evaluate(self, genome, index):
        global maxPsnr

        credits = [0 for _ in range(len(genome)+1)]
        credits[0] = rdbPsnr
        for i in range(1, len(genome)+1):
            _model = GenerateModel(genome[:i], image_size, scale_factor).to(device)
            self.train(_model)
            psnr_, loss = self.test(_model)
            if(maxPsnr[len(genome)] < psnr_):
                PATH = ''
                PATH = PATH.join(genome)
                PATH = str(round(psnr_.item(), 3)) + '-' + PATH + '.pth'
                PATH = os.path.join('parameters', PATH)
                maxPsnr[len(genome)] = psnr_
                torch.save(_model.state_dict(), PATH)
            credits[i] = psnr_
        self.population[index]['fitness'] = psnr_
        print(f'Population : {index+1}, Psnr : {psnr_}')
        credits = [credits[i] - credits[i-1] for i in range(len(credits)-1, 0, -1)]
        return credits[1:]

    def evolution(self):
        global rdbPsnr

        alpha = 0.9
        epsilon = 0.001
        print('Loading Data...')
        self.loadData()
        print('Data Loaded')
        _model = GenerateModel(['r'], image_size, scale_factor).to(device)
        self.train(_model)
        rdbPsnr, loss = self.test(_model)
        

        for generation in range(self.generations):
            print('Generation :', generation+1)
            for i, p in enumerate(self.population):
                if len(p['genome']) < n_max_blocks:
                    p['genome'].append(random.choice(blocks))
                credits = self.evaluate(p['genome'], i)
                for i in range(len(credits)):
                    self.BlockCreditMatrix[p['genome'][i]][i] = alpha*self.BlockCreditMatrix[p['genome'][i]][i] + (1-alpha)*credits[i]

            sorted(self.population, key = lambda p : p['fitness'], reverse = True)
            Elites = self.population[:self.populationSize//2]
            tempPopulation = self.population[:]
            for i in range(self.populationSize//2):
                # mutate
                normalisedCredit = {Type : credit for Type, credit in self.BlockCreditMatrix.items()}
                minCredit = min([pow(self.BlockCreditMatrix[block][j], 2) for j, block in enumerate(Elites[i]['genome'])])
                normalisedCredit = {Type : [x-(minCredit-epsilon) for x in credit] for Type, credit in self.BlockCreditMatrix.items()}
                if random.random() < mutationProb:
                    totalCredit = sum([pow(normalisedCredit[block][j], 2) for j, block in enumerate(Elites[i]['genome'])])
                    for j, block in enumerate(Elites[i]['genome']):
                        selectProb = pow(normalisedCredit[block][j], 2)/totalCredit
                        blockType = blocks[:]
                        blockType.remove(block)
                        p1=pow(normalisedCredit[blockType[0]][j], 2)/totalCredit
                        p2=pow(normalisedCredit[blockType[1]][j], 2)/totalCredit
                        if normalisedCredit[blockType[0]][j]>0 and random.random()<p1:
                            Elites[i]['genome'][j] = blockType[0]
                        elif normalisedCredit[blockType[1]][j]>0 and random.random()<p2:
                            Elites[i]['genome'][j] = blockType[1]
                tempPopulation[i] = Elites[i]

            # crossover
            upto = self.populationSize//2
            for i in range(1, (self.populationSize+1)//2):
                if(upto >= self.populationSize):
                    break
                for j in range(i):
                    if(upto >= self.populationSize):
                        break
                    rand=random.randint(1,self.populationSize-1)

                    crossOver = self.population[i]['genome'][:rand] + self.population[j]['genome'][rand:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
                    if(upto >= self.populationSize):
                        break
                    crossOver = self.population[j]['genome'][:rand] + self.population[i]['genome'][rand:]
                    tempPopulation[upto]['genome'] = crossOver
                    upto += 1
            if(upto != self.populationSize):
                print(f'expected upto be {self.populationSize} but is {upto}')
            self.population = tempPopulation[:]

        return Elites

def run():
    find = GEA(generations, populationSize, mutationProb, elitismNumber)
    final_elites = find.evolution()
    print(f'After evolution over {generations} generations of population of size {populationSize} the Elites are:')
    for e in final_elites:
        print(e)