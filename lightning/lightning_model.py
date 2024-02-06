import pytorch_lightning as pl
import utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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
import model_factory

class PneumoniaModel(pl.LightningModule):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.model, self.img_conf = model_factory.get_model(self.h['model_name'], self.h['classifier_type'], self.h['layers'])
        self.h['image_size'] = self.img_conf['image_size']
        self.h['mean'] = self.img_conf['mean']
        self.h['std'] = self.img_conf['std']
        self.criterion = nn.NLLLoss(weight=self.h['classes_weight'])
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
        probs = torch.exp(outputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        preds = torch.argmax(outputs, dim=1)
        self.test_outputs.append({"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels, 'probs': probs})
        return {"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels, 'probs': probs}

    def on_test_epoch_end(self):
        # compute test metrics
        test_loss_mean = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        test_acc_mean = torch.stack([x["test_acc"] for x in self.test_outputs]).mean()

        self.test_predicted_labels = torch.cat([x["preds"] for x in self.test_outputs], dim=0).cpu().numpy()
        self.test_true_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0).cpu().numpy()
        self.test_probs = torch.cat([x["probs"] for x in self.test_outputs], dim=0).cpu().numpy()[:,1]

        self.test_precision = precision_score(self.test_true_labels, self.test_predicted_labels)
        self.test_recall = recall_score(self.test_true_labels, self.test_predicted_labels)
        self.test_f1 = f1_score(self.test_true_labels, self.test_predicted_labels)
        self.test_acc = test_acc_mean.cpu().numpy()
        self.test_auc = roc_auc_score(self.test_true_labels, self.test_probs)

    def configure_optimizers(self):
        optimizer = self._configure_optimizer()
        scheduler_dic = self._configure_scheduler(optimizer)

        if (scheduler_dic["scheduler"]):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dic
            }            
        else:
            return optimizer

    def _configure_optimizer(self):
        if self.h['optimizer'] == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

        return torch.optim.Adam(self.parameters())
        
    def _configure_scheduler(self, optimizer):
        if self.h['scheduler'] == "":
            return {
                'scheduler': None
            }
        if(self.h['scheduler'] == 'CosineAnnealingLR10'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.h["n_epochs"], eta_min=0.001*0.1)
            return {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        
        if (self.h['scheduler'] == "ReduceLROnPlateau5"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            return {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_acc",
                "strict": True
            }

        print ("Error. scheduler name not valid! '{scheduler_name}'")
        return None











