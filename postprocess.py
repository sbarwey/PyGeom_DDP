"""
Postprocess trained model (no DDP) 
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

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

seed = 42
torch.set_grad_enabled(False)

dataset_dir = './datasets/BACKWARD_FACING_STEP/'

# ~~~~ For making pygeom dataset from VTK
# Get statistics using combined dataset:
path_to_vtk = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_26214_29307_39076_45589.vtk'
data_mean, data_std = bfs.get_data_statistics(
        path_to_vtk,
        multiple_cases = True)



# ~~~~ Load test data
vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_32564.vtk'
path_to_ei = 'datasets/BACKWARD_FACING_STEP/edge_index'
path_to_ea = 'datasets/BACKWARD_FACING_STEP/edge_attr'
path_to_pos = 'datasets/BACKWARD_FACING_STEP/pos'
device_for_loading = 'cpu'

test_dataset, _ = bfs.get_pygeom_dataset_cell_data(
                vtk_file_test, 
                path_to_ei, 
                path_to_ea,
                path_to_pos, 
                device_for_loading,
                time_lag = 2,
                scaling = [data_mean, data_std],
                features_to_keep = [1,2], 
                fraction_valid = 0,  
                multiple_cases = False)

# Remove first snapshot
test_dataset.pop(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load models and Plot losses 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#modelpath = 'saved_models/model_multi_scale.tar'
modelpath = 'saved_models/model_single_scale.tar'
p = torch.load(modelpath)
input_dict = p['input_dict']

# Without top-k: 
input_dict['hidden_channels'] = 8
model = gnn.Multiscale_MessagePassing_UNet(
            in_channels_node = input_dict['in_channels_node'],
            in_channels_edge = input_dict['in_channels_edge'],
            hidden_channels = input_dict['hidden_channels'],
            n_mlp_encode = input_dict['n_mlp_encode'],
            n_mlp_mp = input_dict['n_mlp_mp'],
            n_mp_down = input_dict['n_mp_down'],
            n_mp_up = input_dict['n_mp_up'],
            n_repeat_mp_up = input_dict['n_repeat_mp_up'],
            lengthscales = input_dict['lengthscales'],
            bounding_box = input_dict['bounding_box'],
            act = F.relu, #input_dict['act'],
            interpolation_mode = input_dict['interpolation_mode'],
            name = input_dict['name'])
#model.cuda()
model.eval()
model = torch.jit.script(model)


# Make Prediction
ic_index = 1 # 120
x_new = test_dataset[ic_index].x
t_forward_list = []
for i in range(ic_index,len(test_dataset)):
    print('[%d/%d]' %(i+1, len(test_dataset)))
    data = test_dataset[i]
    print('\tphysical time = %g' %(data.t))
    # Get single step prediction
    t_forward = time.time()
    x_src = model.forward(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
    t_forward = time.time() - t_forward 
    print('\tevaluation time = %g s' %(t_forward))
    t_forward_list.append(t_forward)
    x_new_singlestep = data.x + x_src


np.save('outputs/t_hc_%d.npy' %(input_dict['hidden_channels']), np.array(t_forward_list))




    




