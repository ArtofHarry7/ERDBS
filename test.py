from logging import raiseExceptions
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataLoader import DIV2Kdataset
import psnr
from model import GenerateModel, generateGenome, generatePopulation, getModels
from utility import device, paths
import os

maxPsnr = 0

n_epochs = 1000
image_size = 64
scale_factor = 2
batch_size = 16
learning_rate = 0.0001

nConv = [4, 6, 8]
nChannel = [16, 24, 32, 48, 64]
nRecursion = {1, 2, 3, 4}

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(image_size),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(image_size*scale_factor),
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

            return avgPsnr, avgLoss

def run(test_data, genome):
    print(f'Testing {test_data} images on model woth block sequence:')
    print(list(genome))
    Test = TEST()
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