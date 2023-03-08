"""
Postprocess trained model (no DDP) 
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np

import time
import torch
import torch.utils.data
import torch.distributions as tdist 

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig
Tensor = torch.Tensor

# PyTorch Geometric
import torch_geometric

# Models
import models.cnn as cnn 
import models.gnn as gnn 

# Data preparation
import dataprep.unstructured_mnist as umnist
import dataprep.backward_facing_step as bfs






# Load model 
a = torch.load('saved_models/model.tar')



# Plot losses:
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(a['loss_hist_train'])
ax.plot(a['loss_hist_test'])
#ax.set_yscale('log')
plt.show(block=False)




