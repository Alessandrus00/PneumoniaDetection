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

# Plot the confusion matrix on test data of the best model
class PlotConfusionMatrixCallback(pl.Callback):
    def on_test_end(seld, trainer, pl_module):
        cm = confusion_matrix(pl_module.test_true_labels, pl_module.test_predicted_labels)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NORMAL", "PNEUMONIA"])
        disp.plot()
        plt.show()