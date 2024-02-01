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

# setting up the directories for train, val and test
data_dir = "chest-xray-pneumonia/chest_xray"
train_dir = data_dir + "/train/"
val_dir = data_dir + "/val/"
test_dir = data_dir + "/test/"

class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, h):
        super().__init__()
        self.h = h

    def setup(self, stage=None):
        transforms_train = T.Compose([
            T.RandomRotation(20),  # Randomly rotate the image within a range of (-20, 20) degrees
            T.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally with 50% probability
            T.RandomResizedCrop(size=(299,299), scale=(0.8, 1.0)),  # Randomly crop the image and resize it
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly change the brightness, contrast, saturation, and hue
            T.RandomApply([T.RandomAffine(0, translate=(0.1, 0.1))], p=0.5),  # Randomly apply affine transformations with translation
            T.RandomApply([T.RandomPerspective(distortion_scale=0.2)], p=0.5),
            T.Resize(size=(self.h["image_size"], self.h["image_size"])),
            T.ToTensor(),
            T.Normalize(mean=self.h['mean'], std=self.h['std'])
          ])

        transforms_val = T.Compose([
            T.Resize(size=(self.h["image_size"], self.h["image_size"])),
            T.ToTensor(),
            T.Normalize(mean=self.h['mean'], std=self.h['std']),
        ])

        train_filenames, val_filenames = utilities.split_file_names(train_dir, self.h['val_split'])

        self.train_set = datasets.ImageFolder(train_dir, transform=transforms_train, is_valid_file=lambda x: x in train_filenames)
        self.val_set = datasets.ImageFolder(train_dir, transform=transforms_val, is_valid_file=lambda x: x in val_filenames)
        self.test_set = datasets.ImageFolder(test_dir, transform=transforms_val)

    def train_dataloader(self):
        sampler = utilities.create_weighted_sampler(self.train_set)
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.h["batch_size"], sampler=sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.h["batch_size"])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.h["batch_size"])