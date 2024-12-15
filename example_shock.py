from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import torch
import os 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import models.gnn as gnn 
import dataprep.shock as shk

torch.manual_seed(122)
np.random.seed(122)
SMALL = 1e-10

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a random model, and evaluate on a snapshot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path_to_data = "./datasets/GNN_MeshCent_Field_data_v01_train"
device_for_loading = "cpu"
print('Loading dataset...')
train_list, valid_list = shk.get_pygeom_dataset(
        path_to_data = path_to_data,
        device_for_loading = device_for_loading,
        fraction_valid = 0.0)
print('Done loading dataset.')

data = train_list[503]

# Create 3D scatter plot
# ~~~~ # fig = plt.figure()
# ~~~~ # ax = fig.add_subplot(111, projection='3d')
# ~~~~ # 
# ~~~~ # # Scatter plot with color mapping to 'value'
# ~~~~ # scatter = ax.scatter(data.pos[:,0], data.pos[:,1], data.pos[:,2], c=data.x, cmap='viridis', s=50)  # s is marker size
# ~~~~ # 
# ~~~~ # # Add color bar to show mapping of 'value'
# ~~~~ # cbar = plt.colorbar(scatter, ax=ax)
# ~~~~ # cbar.set_label('Value')
# ~~~~ # 
# ~~~~ # # Set axis labels
# ~~~~ # ax.set_xlabel('X Position')
# ~~~~ # ax.set_ylabel('Y Position')
# ~~~~ # ax.set_zlabel('Z Position')
# ~~~~ # 
# ~~~~ # # Show plot
# ~~~~ # plt.show(block=False)

sample = train_list[0]

# Create model 
device = 'cpu'

# ~~~~ Baseline model settings (no top-k pooling) 
# 1 MMP level (no graph coarsening), and 0 top-k levels 
input_node_channels = sample.x.shape[1]
input_edge_channels = sample.edge_attr.shape[1]
hidden_channels = 64 # embedded node/edge dimensionality   
output_node_channels = input_node_channels
n_mlp_hidden_layers = 2 # size of MLPs
n_mmp_layers = 2 # number of MMP layers per topk level.  
n_messagePassing_layers = 4 # number of message passing layers in each processor block 
max_level_mmp = 0 # maximum number of MMP levels. "1" means only single-scale operations are used. 
max_level_topk = 0 # maximum number of topk levels.
rf_topk = 16 # topk reduction factor 
""" 
if n_mmp_layers=2 and max_level_topk=0, we have this:
    (2 x Down MMP) 0 ---------------> output
"""


# # ~~~~ Interpretable model settings (with top-k pooling)
# # 1 MMP level (no graph coarsening), and 1 top-k level 
# input_node_channels = sample.x.shape[1]
# input_edge_channels = sample.edge_attr.shape[1]
# hidden_channels = 64 # embedded node/edge dimensionality   
# output_node_channels = input_node_channels
# n_mlp_hidden_layers = 2 # size of MLPs
# n_mmp_layers = 1 # number of MMP layers per topk level.  
# n_messagePassing_layers = 4 # number of message passing layers in each processor block 
# max_level_mmp = 0 # maximum number of MMP levels. "1" means only single-scale operations are used. 
# max_level_topk = 1 # maximum number of topk levels.
# rf_topk = 16 # topk reduction factor 
# """ 
# if n_mmp_layers=1 and max_level_topk=1, we have this:
#     (1 x Down MMP) 0 ---------------> (1 x Up MMP) 0 ---> output
#             |                               | 
#             |                               |
#             |----> (1 x Down MMP) 1  ------>|
# """


# get l_char -- characteristic edge lengthscale used for graph coarsening (not used when max_level_mmp=1) 
edge_attr = sample.edge_attr
mean_edge_length = edge_attr[:,2].mean()
l_char = mean_edge_length

model = gnn.TopkMultiscaleGNN( 
            input_node_channels,
            input_edge_channels,
            hidden_channels,
            output_node_channels,
            n_mlp_hidden_layers,
            n_mmp_layers,
            n_messagePassing_layers,
            max_level_mmp,
            l_char,
            max_level_topk,
            rf_topk,
            name='gnn')

model.to(device)

# Evaluate model
model.eval()
data = train_list[0]
x_scaled = (data.x - data.data_mean)/(data.data_std + SMALL) # scaled input
x_src, mask = model(x_scaled, data.edge_index, data.pos, data.edge_attr, data.batch)
x_new = x_scaled + x_src


