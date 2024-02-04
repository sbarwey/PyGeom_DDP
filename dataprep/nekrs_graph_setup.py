import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
from typing import Optional, Union, Callable, List, Tuple

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

def get_stats(x_batch: Tensor) -> Tuple[Tensor, Tensor]:
    x_batch_mean = torch.mean(x_batch, dim=1)
    x_batch_std = torch.std(x_batch, dim=1)
    return x_batch_mean, x_batch_std 

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
                       fraction_valid : Optional[float] = 0.1) -> Tuple[List,List,List,List]:
    t_load = time.time()
   
    print('Loading data and making pygeom dataset...')
    edge_index = np.loadtxt(edge_index_path, dtype=NP_INT).T 
    node_element_ids = np.loadtxt(node_element_ids_path, dtype=NP_INT)
    global_ids = np.loadtxt(global_ids_path, dtype=NP_INT)
    pos = np.loadtxt(pos_path, dtype=NP_FLOAT)
    x = np.loadtxt(data_x_path, dtype=NP_FLOAT)
    y = np.loadtxt(data_y_path, dtype=NP_FLOAT)

    # Retain only N_gll = Np*Ne elements
    N_gll = pos.shape[0]
    x = x[:N_gll, :]
    y = y[:N_gll, :]
    print('N_gll: ', N_gll)

    # Make tensor 
    edge_index = torch.tensor(edge_index)
    node_element_ids = torch.tensor(node_element_ids)
    global_ids = torch.tensor(global_ids)
    pos = torch.tensor(pos, dtype=TORCH_FLOAT)
    x = torch.tensor(x, dtype=TORCH_FLOAT)
    y = torch.tensor(y, dtype=TORCH_FLOAT)

    # Coalesce 
    edge_index = utils.coalesce(edge_index) 

    # un-batch elements 
    pos_tuple = utils.unbatch(pos, node_element_ids, dim=0)
    x_tuple = utils.unbatch(x, node_element_ids, dim=0)
    y_tuple = utils.unbatch(y, node_element_ids, dim=0)
    global_ids_tuple = utils.unbatch(global_ids, node_element_ids, dim=0)

    # get element-local stats -- vector per element   
    x_mean_element, x_std_element = get_stats(torch.stack(x_tuple))
    # y_rms_element = get_rms(torch.stack(y_tuple))

    # get element lengthscale -- single scalar per element  
    lengthscale_element = get_element_lengthscale(torch.stack(pos_tuple))

    # # add ln(rms) to the input data (x) 
    # x_rms = x_rms_element[node_element_ids]
    # x_rms = torch.log(x_rms) # take log -- account for range 
    # x = torch.cat([x, x_rms], dim=1)
    # x_tuple = utils.unbatch(x, node_element_ids, dim=0)

    # Get training data statistics 
    n_samples = x.shape[0]
    x_mean = x.mean(dim=0, keepdim=False)
    x_var = x.var(dim=0, keepdim=False)
    y_mean = y.mean(dim=0, keepdim=False)
    y_var = y.var(dim=0, keepdim=False) 
    data_mean = [x_mean, y_mean] 
    data_var = [x_var, y_var]

    # Train / valid split 
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
        n_nodes_i = x_tuple[i].shape[0]
        data_temp = Data( x = x_tuple[i], 
                          y = y_tuple[i],
                          x_mean = x_mean_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1), 
                          x_std = x_std_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1),
                          L = lengthscale_element[i],
                          pos = pos_tuple[i],
                         pos_norm = pos_tuple[i]/lengthscale_element[i],
                         edge_index = edge_index, 
                         global_ids = global_ids_tuple[i])
        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)

    data_valid_list = []
    for i in idx_valid:
        data_temp = Data( x = x_tuple[i], 
                          y = y_tuple[i], 
                         x_mean = x_mean_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1),
                         x_std = x_std_element[i].unsqueeze(0).repeat(x_tuple[i].shape[0], 1),
                         L = lengthscale_element[i],
                          pos = pos_tuple[i],
                         pos_norm = pos_tuple[i]/lengthscale_element[i],
                         edge_index = edge_index, 
                         global_ids = global_ids_tuple[i])
        data_temp = data_temp.to(device_for_loading)
        data_valid_list.append(data_temp)

    t_load = time.time() - t_load 
    print('\ttook %g sec' %(t_load))
    return data_train_list, data_valid_list, data_mean, data_var, n_samples
