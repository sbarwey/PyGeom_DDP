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
        fraction_valid : Optional[float] = 0.1) -> tuple[list,list]:

    print("in dataprep")
    print(path_to_data)

    files = os.listdir(path_to_data)
    files.remove(".DS_Store")
    data_list = []
    for i in range(len(files)):
        temp = np.loadtxt(path_to_data + "/" + files[i], skiprows=1)
        data_list.append( temp )
    data = np.stack(data_list)
    data_full = torch.tensor(data[:,:,-1:], dtype=torch.float32)
    pos = torch.tensor(data[0,:,:3], dtype=torch.float32)

    # Training data stats 
    data_mean = data_full.mean(dim=(0,1))
    data_std = data_full.std(dim=(0,1))

    # Get edge index using KNN graph
    print(f"pos shape: {pos.shape}")
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

    # Normalize edge attributes 
    max_dist = edge_attr[:,-1].max()
    edge_attr = edge_attr / max_dist

    # Get time indices
    n_snaps = data_full.shape[0]
    data_x_idx = []
    data_y_idx = []
    for i in range(n_snaps):
        data_x_idx.append(i)
        data_y_idx.append(i)

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
        data_y = data_full[idx_y]
        
        data_temp = Data(x = data_x,
                         y = data_y,
                         edge_index = edge_index,
                         edge_attr = edge_attr,
                         pos = pos,
                         bounding_box = [pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()],
                         data_mean = data_mean.unsqueeze(0),
                         data_std = data_std.unsqueeze(0)
                         )

        data_temp = data_temp.to(device_for_loading)
        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    return data_train_list, data_valid_list
