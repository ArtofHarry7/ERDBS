from torch.utils.data import Dataset
import os
from skimage import io
import numpy as np
import math

class DIV2Kdataset(Dataset):
    def __init__(self, lr, hr, train_transform=None, test_transform=None):
        self.lr = self.loadData(lr, train_transform)
        self.hr = self.loadData(hr, test_transform)

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, index):
        return (self.lr[index], self.hr[index])

    def loadData(self, path, transform):
        data = []
        images = os.listdir(path)
        images.sort()
        skip = math.ceil(len(images)/100)
        for i, image_name in enumerate(images):
            image_path = os.path.join(path, image_name)
            image = io.imread(image_path)

            if transform:
                image = transform(image)

            data.append(image)

            if (i+1)%skip == 0:
                print('#', end='')
        print()
        return data
