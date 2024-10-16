"""
Prepares PyGeom data for speedy data  
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import os,time,sys 
import numpy as np

import torch 
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn
import torch_geometric.transforms as transforms


def get_pygeom_dataset(
        path_to_data : str, 
        device_for_loading : str,
        time_lag : Optional[int] = 1,
        fraction_valid : Optional[float] = 0.1) -> tuple[list,list]:

    a = np.load(path_to_data)
    field_names = a.files
    data_temp = a[field_names[0]]
    time_vec = torch.arange(data_temp.shape[0], dtype=torch.float32)
    img = data_temp[0]
    Nx = img.shape[0]
    Ny = img.shape[1]

    # Create full data 
    data_full = []
    for f in field_names:
        data_full.append( torch.tensor( a[f].reshape(-1, Nx*Ny) ).unsqueeze(-1) )
    data_full = torch.cat(data_full, dim=-1)

    # Training data stats 
    data_mean = data_full.mean(dim=(0,1))
    data_std = data_full.std(dim=(0,1))

    # Get position
    img = data_temp[0]
    x_lim = np.linspace(0,1,Nx)
    y_lim = np.linspace(0,1,Ny)
    X,Y = np.meshgrid(x_lim, y_lim, indexing='ij')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    #pos = torch.cat( (X.reshape(-1,1), Y.reshape(-1,1)), axis=1 )
    pos = torch.cat( (Y.reshape(-1,1), -X.reshape(-1,1) + 1), axis=1 ) # rotated

    # Get edge index using KNN graph
    edge_index = tgnn.knn_graph(pos, k=10)
    edge_index = utils.to_undirected(edge_index)

    # Get edge attrtorch_geometric.utils
    data_ref = Data( pos = pos, edge_index = edge_index )
    cart = transforms.Cartesian(norm=False, max_value = None, cat = False)
    dist = transforms.Distance(norm = False, max_value = None, cat = True)

    # populate edge_attr
    data_ref = cart(data_ref) # adds cartesian/component-wise distance
    data_ref = dist(data_ref) # adds euclidean distance

    # extract edge_attr
    edge_attr = data_ref.edge_attr

    # Eliminate duplicate edges
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # Get time indices
    n_snaps = data_full.shape[0] - time_lag
    data_x_idx = []
    data_y_idx = []
    for i in range(n_snaps):
        data_x_idx.append(i)
        if time_lag == 0:
            y_temp = i
        else:
            y_temp = []
            time_temp = []
            for t in range(1, time_lag+1):
                y_temp.append(i+t)
        data_y_idx.append(y_temp)

    # Train / valid split
    if fraction_valid > 0:
        # How many total snapshots to extract
        n_full = n_snaps
        n_valid = int(np.floor(fraction_valid * n_full))

        # Get validation set indices
        idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

        # Get training set indices
        idx_train = np.array(list(set(list(range(n_full))) - set(list(idx_valid))))

        n_train = len(idx_train)
        n_valid = len(idx_valid)
    else:
        n_full = n_snaps
        n_valid = 0
        n_train = n_full
        idx_train = list(range(n_train))
        idx_valid = []

    idx_train_mask = np.zeros(n_snaps, dtype=int)
    idx_train_mask[idx_train] = 1

    data_train_list = []
    data_valid_list = []
    for i in range(n_snaps):
        idx_x = data_x_idx[i]
        idx_y = data_y_idx[i]
        data_x = data_full[idx_x]
        time_x = time_vec[idx_x]
        if time_lag == 0:
            data_y = data_full[idx_y]
            time_y = time_vec[idx_y]
        else:
            data_y = []
            time_y = []
            for t in range(time_lag):
                data_y.append(data_full[idx_y[t]])
                time_y.append(time_vec[idx_y[t]])

        data_temp = Data(x = data_x,
                         y = data_y,
                         edge_index = edge_index,
                         edge_attr = edge_attr,
                         pos = pos,
                         bounding_box = [pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()],
                         data_mean = data_mean.unsqueeze(0),
                         data_std = data_std.unsqueeze(0),
                         t_x = time_x,
                         t_y = time_y,
                         field_names = field_names,
                         field_shape = img.shape)

        data_temp = data_temp.to(device_for_loading)
        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    return data_train_list, data_valid_list
