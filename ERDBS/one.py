import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import os
import cv2
from PIL import Image

# %matplotlib inline


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 


# data
# mean = []
# std = []

train_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(90),
    transforms.ToTensor(),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

train_hr_path = './PyTorchPractice__/ERDBS/Data/DIV2K/train/DIV2K_train_HR'
train_lr_path = './PyTorchPractice__/ERDBS/Data/DIV2K/train/DIV2K_train_LR_bicubic/X2'
test_hr_path = './PyTorchPractice__/ERDBS/Data/DIV2K/test/DIV2K_test_HR'
test_lr_path = './PyTorchPractice__/ERDBS/Data/DIV2K/test/DIV2K_test_LR_bicubic/X2'

# train_lr_dataset = torchvision.datasets.ImageFolder(root = train_lr_path, transform=train_transform)
# test_lr_dataset = torchvision.datasets.ImageFolder(root = test_lr_path, transform=test_transform)

# print(train_dataset[0][1])

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break

# train_hr = np.load(train_hr_path)
# train_lr = np.load(train_lr_path)
# test_hr = np.load(test_hr_path)
# test_lr = np.load(test_lr_path)

test_path = './PyTorchPractice__/ERDBS/Data/DIV2K/train/DIV2K_train_LR_bicubic/X2/0001x2.png'
test_image = Image.open(test_path)
np_image = np.array(test_image)
print(test_image.format, test_image.size, np_image)
np_image = np.reshape(np_image, (3, 1020, 702), order='A')
print(np_image)

# def load_data(hr_path, lr_path):
#     labels = []
#     images = []
#     Files = os.listdir(image_dir)
#     for File in Files:
#         path = os.path.join(image_dir, File)
#         image = cv2.imread(path)
#         # image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
#         images.append(image)
#     return images

# train_hr = load_data(train_hr_path)
# train_lr = load_data(train_lr_path)
# test_hr = load_data(test_hr_path)
# test_lr = load_data(test_lr_path)
# print(len(train_hr), len(train_hr), len(train_hr), len(train_hr))

class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.transform(self.x[index])

    transform = transforms.Compose([
        # transforms.Resize(64),
        transforms.ToTensor(),
    ])

# batch_size = 16
# cropped_dataset = dataset(ims=X_train)
# train_dl = DataLoader(cropped_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
# show_batch(train_dl)