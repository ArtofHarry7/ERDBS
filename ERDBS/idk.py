# from re import A
import torch
import os
import RDBs
from utility import device
from model import GenerateModel
import torch.nn as nn
import random
import imp

# print(torch.cuda.is_available()) 
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))
# print(torch.cuda.memory_allocated())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# pwd = os.getcwd()
# train_hr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR")
# train_lr_path = os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2")
# test_hr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR")
# test_lr_path = os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2")
# check_hr_path = os.path.join(pwd, "Data", "DIV2K", "check", "HR")
# check_lr_path = os.path.join(pwd, "Data", "DIV2K", "check", "LR")


# print(train_lr_path)
# genome = ['g', 'g', 's', 'g', 'g', 'c', 'c', 's', 'g', 'g', 'c', 'g']
# _model = GenerateModel(genome).to(device)
# # print((_model.parameters))
# # for parameter in _model.parameters():
# #     print(parameter.shape)
# x = torch.rand([3, 3, 32, 32]).to(device)
# y = _model(x)
# print(y.shape)
# print(device)
# model = RDBs.SRDB(64, 64, 6).to(device)
# model = RDBs.SRDB(64, 64, 6).to(device)
# model = RDBs.GRDB(64, 64, 6, 4).to(device)
# linear = nn.ModuleList([self.RDB[rdb] for rdb in self.pattern])
# model = GenerateModel(genome)
# x = torch.rand(1, 64, 64, 64).to(device)
# y = model(x)
# print(y.shape, 'done')
# y = ['a', 'r', 't', 'i']
# path = ''
# path = path.join(y)
# print(path)
# x = torch.tensor(46.858).to(device)
# path += str(round(x.item(), 3))
# print(path)
# path = os.path.join('parameters', path)
# print(path)

# a = 5
# print(a/2, a//2)

# r = random.random()
# s = random.random()
# print(r, s)
# print([i for i in range(8, 0, -1)])


# a = {
#     's' : [4, 5, 9, 1],
#     'g' : [2, 3, 8, 4],
#     'c' : [2, 0, 7, 3]
#     }
# print(a)
# b = {Type : credit for Type, credit in a.items()}
# # b['s'][1] = 6
# b = {Type : [x-2 for x in credit] for Type, credit in b.items()}
# print(a, b)
a = [4, 5, 6, 3]
print(min(a))