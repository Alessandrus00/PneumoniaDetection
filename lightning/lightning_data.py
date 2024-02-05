import pytorch_lightning as pl
import utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm.auto import tqdm
import time
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pprint
import timm
import pytorch_lightning as pl
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

# setting up the directories for train, val and test
data_dir = "chest-xray-pneumonia/chest_xray"
train_dir = data_dir + "/train/"
val_dir = data_dir + "/val/"
test_dir = data_dir + "/test/"

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, is_valid_file=None):
        self.dataset = datasets.ImageFolder(root, is_valid_file=is_valid_file)
        self.transform = transform
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image=np.array(image))["image"] / 255.0
        return image, label

    def __len__(self):
        return len(self.dataset)


class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, h):
        super().__init__()
        self.h = h

    def setup(self, stage=None):
        
        if self.h['augmentation'] == 'normal':
            # normal augmentation
            data_transforms_train_alb = A.Compose([
                A.Rotate(limit=20),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1), 
                A.Resize(height=self.h["image_size"], width=self.h["image_size"]),
                A.Normalize(mean=list(self.h['mean']), std=list(self.h['std'])),
                ToTensorV2()
            ])
        else:
            # strong augmentation
            data_transforms_train_alb = A.Compose([
                A.Rotate(limit=20),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
                A.Resize(height=self.h["image_size"], width=self.h["image_size"]),
                A.Normalize(mean=list(self.h['mean']), std=list(self.h['std'])),
                ToTensorV2()
            ])

        data_transforms_val_alb = A.Compose([
            A.Resize(self.h["image_size"], self.h["image_size"]),
            A.Normalize(mean=list(self.h['mean']), std=list(self.h['std'])),
            ToTensorV2(),
        ])

        train_filenames, val_filenames = utilities.split_file_names(train_dir, self.h['val_split'])

        self.train_set = CustomImageFolder(train_dir, transform=data_transforms_train_alb, is_valid_file=lambda x: x in train_filenames)
        self.val_set = CustomImageFolder(train_dir, transform=data_transforms_val_alb, is_valid_file=lambda x: x in val_filenames)
        self.test_set = CustomImageFolder(test_dir, transform=data_transforms_val_alb)

    def train_dataloader(self):
        sampler = utilities.create_weighted_sampler(self.train_set)
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.h["batch_size"], sampler=sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.h["batch_size"])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.h["batch_size"])