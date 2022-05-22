import torch
import os

n_min_blocks = 5
n_max_blocks = 20


pwd = os.getcwd()
paths = {
    'lr' : {
        'check' : os.path.join(pwd, "Data", "DIV2K", "check", "LR"),
        'b100' : os.path.join(pwd, "Data", "benchmark", "B100", "LR_bicubic", "X2"),
        'set5' : os.path.join(pwd, "Data", "benchmark", "Set5", "LR_bicubic", "X2"),
        'set14' : os.path.join(pwd, "Data", "benchmark", "Set14", "LR_bicubic", "X2"),
        'urban100' : os.path.join(pwd, "Data", "benchmark", "Urban100", "LR_bicubic", "X2"),
        'div2k' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2"),
        'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_LR_bicubic", "X2"),
        'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_LR_bicubic", "X2"),
    },
    'hr' : {
        'check' : os.path.join(pwd, "Data", "DIV2K", "check", "HR"),
        'b100' : os.path.join(pwd, "Data", "benchmark", "B100", "HR"),
        'set5' : os.path.join(pwd, "Data", "benchmark", "Set5", "HR"),
        'set14' : os.path.join(pwd, "Data", "benchmark", "Set14", "HR"),
        'urban100' : os.path.join(pwd, "Data", "benchmark", "Urban100", "HR"),
        'div2k' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR"),
        'train' : os.path.join(pwd, "Data", "DIV2K", "train", "DIV2K_train_HR"),
        'test' : os.path.join(pwd, "Data", "DIV2K", "validation", "DIV2K_valid_HR"),
    }
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')