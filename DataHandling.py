import torch.nn as nn
import torch
import numpy as np
import os
import cv2 as cv
import torchvision
import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.utils as vutils
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
#import scipy.io as io
from torchsummary import summary
import glob
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import re, PIL, math
from PIL import Image
import random, time
from einops.layers.torch import Reduce, Rearrange
from torchvision import models
from sklearn.metrics import f1_score
from torch.nn import init

class CustomDataset(Dataset):
    def __init__(self, train_type = 'train', transform = None):
        super(CustomDataset, self).__init__()
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
            #self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.dataset = torchvision.datasets.VOCSegmentation('.', image_set = train_type,
                                               transform = None,
                                               target_transform = None,
                                               download = True)
    def __getitem__(self, i):
        image = self.dataset[i][0]
        target = self.dataset[i][1]
        c = np.array(target) == np.max(np.array(target))
        s = np.array(target) != np.min(np.array(target))
        contour = c.astype(np.uint8)
        saliency = s.astype(np.uint8)
        contour = Image.fromarray(np.squeeze(contour))
        saliency = Image.fromarray(np.squeeze(saliency))
        seed = np.random.randint(212564736)
        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:
            image, target = self.transform(image), self.transform(target)

        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:
            contour = self.transform(contour)
            #
        random.seed(seed)
        torch.manual_seed(seed)
        if self.transform is not None:
            saliency = self.transform(saliency)

        contour /= torch.max(contour)
        saliency /= torch.max(saliency)
        
        return image, saliency, contour

    def __len__(self):
        return len(self.dataset)

def get_augmented_images(img, rotation = 0, diff = 1):
    res = []
    res.append(img.transpose(PIL.Image.FLIP_TOP_BOTTOM))
    res.append(img.transpose(PIL.Image.FLIP_LEFT_RIGHT))
    if rotation:
        for i in range(-rotation, rotation + 1, diff):
            res.append(img.rotate(i, resample=Image.BILINEAR))
    else:
        res.append(img.rotate(20, resample=Image.BILINEAR))
        res.append(img.rotate(-20, resample=Image.BILINEAR))
    return res


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return IoU

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class AugmentedDataset(Dataset):
    """docstring for AugmentedDataset"""
    def __init__(self, train_type = True):
        super(AugmentedDataset, self).__init__()
        self.dataset = CustomDataset(train_type = train_type, transform = None)
        self.transf = transforms.ToTensor()
        self.create_data = []
        self.getitem()

    def getitem(self):
        for i in range(len(self.dataset)):
            image = transforms.ToPILImage()(self.dataset[i][0])
            saliency = transforms.ToPILImage()(self.dataset[i][1])
            contour = transforms.ToPILImage()(self.dataset[i][2])
            for i, j, k in zip(get_augmented_images(image, 90, 45), get_augmented_images(saliency, 90, 45), get_augmented_images(contour, 90, 45)):
                self.create_data.append([self.transf(i), self.transf(j), self.transf(k)])

    def __getitem__(self, i):
        return self.create_data[i]

    def __len__(self):
        return len(self.create_data)
