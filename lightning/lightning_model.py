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

class PneumoniaModel(pl.LightningModule):
    def __init__(self, h, model):
        super().__init__()
        self.h = h
        self.model = model
        self.criterion = nn.NLLLoss()
        self.test_outputs = []


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        metrics = {"train_loss": loss, "train_acc": acc*100}
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        metrics = {"val_loss": loss, "val_acc": acc*100}
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=False)
        return metrics

    def on_test_epoch_start(self):
        # initilize test metrics
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        preds = torch.argmax(outputs, dim=1)
        self.test_outputs.append({"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels})
        return {"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels}

    def on_test_epoch_end(self):
        # compute test metrics
        test_loss_mean = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        test_acc_mean = torch.stack([x["test_acc"] for x in self.test_outputs]).mean()

        self.test_predicted_labels = torch.cat([x["preds"] for x in self.test_outputs], dim=0).cpu().numpy()
        self.test_true_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0).cpu().numpy()

        self.test_precision = precision_score(self.test_true_labels, self.test_predicted_labels)
        self.test_recall = recall_score(self.test_true_labels, self.test_predicted_labels)
        self.test_f1 = f1_score(self.test_true_labels, self.test_predicted_labels)
        self.test_acc = test_acc_mean.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _configure_scheduler(self, optimizer):
        return None