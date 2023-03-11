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


import matplotlib.pyplot as plt




# Load model 
a = torch.load('saved_models/model_single_scale.tar')
b = torch.load('saved_models/model_multi_scale.tar')



# Plot losses:
fig, ax = plt.subplots(1,2,sharey=True)
ax[0].plot(a['loss_hist_train'])
ax[0].plot(a['loss_hist_test'])
ax[0].set_yscale('log')
ax[0].set_ylim([1e-3, 1e-1])
ax[0].set_xlim([0,150])

ax[1].plot(b['loss_hist_train'])
ax[1].plot(b['loss_hist_test'])
ax[1].set_yscale('log')
ax[1].set_xlim([0,150])
plt.show(block=False)




