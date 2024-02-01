import pytorch_lightning as pl
import additional.utilities
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

# Plot training/val learning curves for both accuracy and loss over epochs
# the values are restored in case of a failure in the training, thanks to the state dict
class PlotLearningCurvesCallback(pl.Callback):
    def __init__(self):
        self.state = {'train_losses': [], 'train_accs': [], 'val_losses': [], 'val_accs': []}

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss_epoch" in trainer.callback_metrics and 'train_acc_epoch' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics["train_loss_epoch"].cpu().numpy()
            train_acc = trainer.callback_metrics["train_acc_epoch"].cpu().numpy()
            self.state['train_losses'].append(train_loss)
            self.state['train_accs'].append(train_acc)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if "val_loss" in trainer.callback_metrics and "val_acc" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            val_acc = trainer.callback_metrics["val_acc"].cpu().numpy()
            self.state['val_losses'].append(val_loss)
            self.state['val_accs'].append(val_acc)

    def on_fit_end(self, trainer, pl_module):
        # Plot results after trainind ends
        plt.figure(figsize=(20, 6))
        _ = plt.subplot(1,2,1)
        plt.plot(np.arange(trainer.max_epochs) + 1, self.state['train_losses'], 'o-', linewidth=3)
        plt.plot(np.arange(trainer.max_epochs) + 1, self.state['val_losses'], 'o-', linewidth=3)
        _ = plt.legend(['Train', 'Validation'])
        plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Loss')

        _ = plt.subplot(1,2,2)
        plt.plot(np.arange(trainer.max_epochs) + 1, self.state['train_accs'], 'o-', linewidth=3)
        plt.plot(np.arange(trainer.max_epochs) + 1, self.state['val_accs'], 'o-', linewidth=3)
        _ = plt.legend(['Train', 'Validation'])
        plt.grid('on'), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.show()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()