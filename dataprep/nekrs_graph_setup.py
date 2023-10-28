import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
from typing import Optional, Union, Callable

Tensor = torch.Tensor 
TORCH_FLOAT = torch.float32
NP_FLOAT = np.float32
TORCH_INT = torch.int64
NP_INT = np.int64


def get_rms(x_batch: Tensor) -> Tensor:
    u_var = x_batch.var(dim=1, keepdim=True)
    tke = 0.5*u_var.sum(dim=2)
    u_rms = torch.sqrt(tke / 1.5)
    return u_rms 

def get_element_lengthscale(pos_batch: Tensor) -> Tensor:
    pos_min = pos_batch.min(dim=1)[0]
    pos_max = pos_batch.max(dim=1)[0]
    return torch.norm(pos_max - pos_min, p=2, dim=1)

def get_pygeom_dataset(data_x_path: str,
                       data_y_path: str,
                       edge_index_path: str,
                       node_element_ids_path: str,
                       global_ids_path: str,
                       pos_path: str, 
                       device_for_loading : Optional[str] = 'cpu',
                       fraction_valid : Optional[float] = 0.1) -> tuple[list,list]:
    t_load = time.time()
   
    print('Loading data and making pygeom dataset...')
    edge_index = np.loadtxt(edge_index_path, dtype=NP_INT).T 
    node_element_ids = np.loadtxt(node_element_ids_path, dtype=NP_INT)
    global_ids = np.loadtxt(global_ids_path, dtype=NP_INT)
    pos = np.loadtxt(pos_path, dtype=NP_FLOAT)
    x = np.loadtxt(data_x_path, dtype=NP_FLOAT)
    y = np.loadtxt(data_y_path, dtype=NP_FLOAT)

    # Make tensor 
    edge_index = torch.tensor(edge_index)
    node_element_ids = torch.tensor(node_element_ids)
    global_ids = torch.tensor(global_ids)
    pos = torch.tensor(pos)
    x = torch.tensor(x)
    y = torch.tensor(y)

    # Coalesce 
    edge_index = utils.coalesce(edge_index) 

    # un-batch elements 
    pos_tuple = utils.unbatch(pos, node_element_ids, dim=0)
    x_tuple = utils.unbatch(x, node_element_ids, dim=0)
    y_tuple = utils.unbatch(y, node_element_ids, dim=0)
    global_ids_tuple = utils.unbatch(global_ids, node_element_ids, dim=0)

    # get rms 
    x_rms_element = get_rms(torch.stack(x_tuple))
    y_rms_element = get_rms(torch.stack(y_tuple))
    x_rms = x_rms_element[node_element_ids]
    y_rms = y_rms_element[node_element_ids]

    # get element lengthscale 
    lengthscale_element = get_element_lengthscale(torch.stack(pos_tuple))

    # add ln(rms) to the input data (x) 
    x_rms = torch.log(x_rms) # take log -- account for range 
    x = torch.cat([x, x_rms], dim=1)
    x_tuple = utils.unbatch(x, node_element_ids, dim=0)

    # Get training data statistics 
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    y_std = y.std(dim=0, keepdim=True) + 1e-15
    data_mean = [x_mean, y_mean] 
    data_std = [x_std, y_std]

    # Train / valid split 
    print('number of elements (graph samples): %d' %(len(x_tuple)))
        
    n_snaps = len(pos_tuple)

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

    data_train_list = []
    for i in idx_train:
        data_temp = Data( x = x_tuple[i], 
                          y = y_tuple[i],
                          x_rms = x_rms_element[i],
                         y_rms = y_rms_element[i],
                          L = lengthscale_element[i],
                          pos = pos_tuple[i],
                         edge_index = edge_index, 
                         global_ids = global_ids_tuple[i])
        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)

    data_valid_list = []
    for i in idx_valid:
        data_temp = Data( x = x_tuple[i], 
                          y = y_tuple[i], 
                          x_rms = x_rms_element[i],
                         y_rms = y_rms_element[i],
                         L = lengthscale_element[i],
                          pos = pos_tuple[i],
                         edge_index = edge_index, 
                         global_ids = global_ids_tuple[i])
        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)

    t_load = time.time() - t_load 
    print('\ttook %g sec' %(t_load))
    return data_train_list, data_valid_list, data_mean, data_std 
