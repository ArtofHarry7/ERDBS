from random import shuffle
import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from dataLoader import DIV2Kdataset
import numpy as np
import psnr
import PIL
from model import GenerateModel, generateGenome, generatePopulation, getModels
from utility import n_blocks, device
import math

n_epochs = 10

pwd = os.getcwd()
train_hr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR")
train_lr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2")
test_hr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR")
test_lr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2")
check_hr_path = os.path.join(pwd, "Data", "DIV2K", "check", "HR")
check_lr_path = os.path.join(pwd, "Data", "DIV2K", "check", "LR")

image_size = 64
batch_size = 10

train_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.CenterCrop(64),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(90),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(128),
    # transforms.Resize((128, 128)),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

nConv = [4, 6, 8]
nChannel = [16, 24, 32, 48, 64]
nRecursion = {1, 2, 3, 4}

# random_genome = generateGenome(n_blocks)
# random_genome = ['c', 'g', 'c', 's', 'g', 's', 'c', 'g'] # 68.8007 0.0108
# random_genome = ['g', 'c', 'g', 'c', 'c', 's', 'g', 'c'] # 67.3925 0.0231
# random_genome = ['c', 'g', 's', 's', 'c', 'c', 'g', 'g'] # 67.4872 0.0137
# random_genome = ['g', 's', 's', 's', 'c', 'c', 'g', 'g'] # 64.3135 0.0303 ----
# random_genome = ['c', 'g', 'g', 's', 's', 'g', 'g', 'c'] # 64.8709 0.0270
# random_genome = ['s', 'c', 's', 'c', 'c', 'g', 'c', 'g'] # 69.8957 0.086
# random_genome = ['g', 'g', 's', 'g', 'c', 's', 'c', 'c'] # 62.0198 0.0517
# random_genome = ['g', 's', 'c', 's', 'c', 'c', 'g', 'c'] # 68.9425 0.0102
# random_genome = ['g', 's', 'c', 's', 'g', 's', 'c', 's'] # 70.9031 0.0070
# random_genome = ['s', 'c', 's', 's', 'g', 'c', 'c', 'c'] # 60.7005 0.0718 ----
# random_genome = ['g', 's', 'c', 'c', 'c', 'c', 'c', 'g'] # 60.9091 0.0690
# random_genome = ['s', 'c', 'g', 'c', 'c', 'c', 'c', 's'] # 61.5300 0.0576
# random_genome = ['g', 's', 'g', 'c', 'c', 's', 'g', 's'] # 70.1977 0.0083
# random_genome = ['c', 's', 's', 'g', 'c', 'c', 'g', 's'] # 64.8731 0.0271 --- ---
random_genome = ['g', 'c', 'c', 'c', 's', 'g', 'g', 's'] # 69.0800 0.0101 ---
# random_genome = ['s', 's', 's', 's', 'g', 's', 'c', 's'] # break

print(random_genome)
model__ = GenerateModel(random_genome)
model__ = model__.to(device)
print(torch.cuda.memory_allocated())

# mseLoss = nn.MSELoss()
l1Loss = nn.L1Loss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model__.parameters(), lr=learning_rate)

def train(n_epochs):
    train_data = DIV2Kdataset(train_lr_path, train_hr_path, train_transform, test_transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=16,
        shuffle=True,
    )
    total_sample = len(train_data)
    n_iteration = math.ceil(total_sample/batch_size)
    for epoch in range(n_epochs):
        for i, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            y = model__(lr)

            psnrValue = psnr.PSNR()
            # loss = mseLoss(y, hr)
            loss = l1Loss(y, hr)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)%10 == 0:
                print(f'epoch [{epoch+1}/{n_epochs}], step [{i+1}/{n_iteration}], loss: {loss.item():.4f}, psnr: {psnrValue(y, hr)}')


avgPsnr = 0
avgLoss = 0
total = 0

def test():
    global avgPsnr
    global avgLoss
    global total
    
    test_data = DIV2Kdataset(test_lr_path, test_hr_path, train_transform, test_transform)

    test_loader = DataLoader(
        dataset=test_data,
    )

    n_total_steps = len(test_loader)
    with torch.no_grad():
        for i, (lr, hr) in enumerate(test_loader):
            lr = lr.to(device)
            hr = hr.to(device)
            y = model__(lr)

            psnrValue = psnr.PSNR()
            # loss = mseLoss(y, hr)
            loss = l1Loss(y, hr)
            
            total += 1
            avgPsnr += psnrValue(y, hr)
            avgLoss += loss
            if (i+1)%10 == 0:
                print(f'step [{i+1}/{n_total_steps}], loss: {loss.item():.4f}, psnr: {psnrValue(y, hr)}')

        avgLoss /= total
        avgPsnr /= total

        test = [x for x in test_loader]

        tra = transforms.ToPILImage()
        eg = model__(test[0][0].to(device))
        im_lr = tra(test[0][0][0])
        im_hr = tra(test[0][1][0])
        img = tra(eg[0])
        im_lr.show()
        im_hr.show()
        img.show()


PATH = 'model.pth'
def train_n_save():
    train(n_epochs=10)

    torch.save(model__.state_dict(), PATH)

def load():
    # model__ = GenerateModel(random_genome)
    model__.load_state_dict(torch.load(PATH))
    test()
    print(avgLoss, avgPsnr)
    print(device)

# train_n_save()
load()
print('done')

# ['g', 'c', 'g', 'c', 'c', 's', 'g', 'c']
