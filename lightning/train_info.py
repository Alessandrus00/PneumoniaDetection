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

# Regularly print training information
class PrintTrainingInfoCallback(pl.Callback):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CPU cores: {os.cpu_count()}, Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CPU cores: {os.cpu_count()}, Device: {device}")

    def setup(self, trainer, pl_module, stage):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip the sanity check
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs

        # Compute time metrics
        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = total_epochs - epoch - 1
        remaining_time = remaining_epochs * avg_time_per_epoch

        # Convert elapsed time in hh:mm:ss
        elapsed_time_hr, rem = divmod(elapsed_time, 3600)
        elapsed_time_min, elapsed_time_sec = divmod(rem, 60)

        # Convert remaining time in hh:mm:ss
        remaining_time_hr, rem = divmod(remaining_time, 3600)
        remaining_time_min, remaining_time_sec = divmod(rem, 60)

        # print time metrics and train/val metrics of current epoch
        print(f'Epoch: {epoch+1:03d}')

        if "train_loss_epoch" in trainer.callback_metrics and 'train_acc_epoch' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics["train_loss_epoch"].cpu().numpy()
            train_acc = trainer.callback_metrics["train_acc_epoch"].cpu().numpy()
            print(f'\tTrain Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        else:
            print(f"Train Loss and/or Acc not available", end="")

        if "val_loss" in trainer.callback_metrics and "val_acc" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            val_acc = trainer.callback_metrics["val_acc"].cpu().numpy()
            print(f'\tVal Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')
        else:
            print(f"Validation Loss and/or Acc not available", end="")

        print(f"\tElapsed time: {elapsed_time_hr:.0f}h {elapsed_time_min:.0f}m {elapsed_time_sec:.0f}s, Remaining Time: {remaining_time_hr:.0f}h {remaining_time_min:.0f}m {remaining_time_sec:.0f}s")
