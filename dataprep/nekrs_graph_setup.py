import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
from typing import Optional, Union, Callable, List, Tuple
from pymech.neksuite import readnek

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

def get_edge_index_coincident(batch, pos, edge_index):
    if batch is None:
        batch = edge_index.new_zeros(pos.size(0))

    pos_unbatch = utils.unbatch(pos, batch)
    ei_coin_unbatch = []
    n_nodes_unbatch = []
    n_nodes_incr = [0] 
    for b in range(batch.max()+1):
        pos = pos_unbatch[b]
        ei_coin = tgnn.radius_graph(pos, r = 1e-9, max_num_neighbors=32)
        n_nodes_unbatch.append(pos.shape[0])
        ei_coin_unbatch.append(ei_coin)
        if b > 0:
            n_nodes_incr.append(n_nodes_unbatch[b] + n_nodes_incr[b-1])

    for b in range(batch.max()+1):
        ei_coin_unbatch[b] = ei_coin_unbatch[b] + n_nodes_incr[b] 

    ei_coin = torch.concat(ei_coin_unbatch, dim=1)
    return ei_coin

def get_pygeom_dataset_pymech(data_x_path: str,
                       data_y_path: str,
                       edge_index_path: str,
                       edge_index_vertex_path: Optional[str] = None,
                       node_weight: Optional[float] = 1.0,
                       device_for_loading : Optional[str] = 'cpu',
                       fraction_valid : Optional[float] = 0.1,
                       n_element_neighbors : Optional[int] = 0) -> Tuple[List,List]:
    t_load = time.time()
   
    print('Loading data and making pygeom dataset...')
    edge_index = np.loadtxt(edge_index_path, dtype=np.int64).T 
    if edge_index_vertex_path: 
        print('Adding p1 connectivity...')
        print('\tEdge index shape before: ', edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=np.int64).T 
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        print('\tEdge index shape after: ', edge_index.shape)
    edge_index = torch.tensor(edge_index)

    n_nodes_per_element = edge_index.max() + 1

    if n_element_neighbors > 0:
        node_max_per_element = edge_index.max()
        n_edges_per_element = edge_index.shape[1]
        edge_index_full = torch.zeros((2, n_edges_per_element*(n_element_neighbors+1)), dtype=edge_index.dtype)
        edge_index_full[:, :n_edges_per_element] = edge_index
        for i in range(1,n_element_neighbors+1):
            start = n_edges_per_element*i
            end = n_edges_per_element*(i+1)
            edge_index_full[:, start:end] = edge_index + (node_max_per_element+1)*i
        edge_index = edge_index_full


    # Load data 
    x_field = readnek(data_x_path)
    y_field = readnek(data_y_path)

    # Train / valid split 
    n_snaps = len(x_field.elem)

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
    
    # Get the element neighborhoods
    if n_element_neighbors > 0:
        Nelements = len(x_field.elem)
        pos_c = torch.zeros((Nelements, 3))
        for i in range(Nelements):
            pos_c[i] = torch.tensor(x_field.elem[i].centroid)
        edge_index_c = tgnn.knn_graph(x = pos_c, k = n_element_neighbors)

    # Get the element masks
    central_element_mask = torch.concat(
            (torch.ones((n_nodes_per_element), dtype=torch.int64),
             torch.zeros((n_nodes_per_element * n_element_neighbors), dtype=torch.int64))
            )
    central_element_mask = central_element_mask.to(torch.bool)

    data_train_list = []
    data_valid_list = []
    for i in range(n_snaps):
        N_i = x_field.elem[i].pos.shape[1]
        pos_x_i = torch.tensor(x_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        pos_y_i = torch.tensor(y_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3]
        vel_x_i = torch.tensor(x_field.elem[i].vel).reshape((3, -1)).T
        vel_y_i = torch.tensor(y_field.elem[i].vel).reshape((3, -1)).T

        x_gll = x_field.elem[i].pos[0,0,0,:]
        dx_min = x_gll[1] - x_gll[0]

        error_max = (pos_x_i - pos_y_i).max()
        rel_error = torch.abs(error_max / dx_min)*100
    
        # Check positions 
        if rel_error > 1e-2:
            print(f"Relative error in positions exceeds 0.01% in element i={i}.")
            sys.exit()
        if (pos_x_i.max() == 0. and pos_x_i.min() == 0.): 
            print(f"Node positions are not stored in {data_x_path}.")
            sys.exit()
        if (pos_y_i.max() == 0. and pos_y_i.min() == 0.): 
            print(f"Node positions are not stored in {data_y_path}.")
            sys.exit()
        pos_i = pos_x_i
        
        # get x_mean and x_std 
        x_mean_element = torch.mean(vel_x_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1) 
        x_std_element = torch.std(vel_x_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1)

        # element lengthscale 
        lengthscale_element = torch.norm(pos_i.max(dim=0)[0] - pos_i.min(dim=0)[0], p=2)

        # node weight 
        nw = torch.ones((vel_x_i.shape[0], 1)) * node_weight

        # Get the element neighbors for the input  
        if n_element_neighbors > 0:
            send = edge_index_c[0,:]
            recv = edge_index_c[1,:]
            nbrs = send[recv == i]

            pos_x_full = [pos_x_i]
            vel_x_full = [vel_x_i]
            for j in nbrs:
                pos_x_full.append( torch.tensor(x_field.elem[j].pos).reshape((3, -1)).T )
                vel_x_full.append( torch.tensor(x_field.elem[j].vel).reshape((3, -1)).T )
            pos_x_full = torch.concat(pos_x_full)
            vel_x_full = torch.concat(vel_x_full)

            # reset pos 
            pos_i = pos_x_full
            vel_x_i = vel_x_full


        # create data 
        data_temp = Data( x = vel_x_i.to(dtype=TORCH_FLOAT), 
                          y = vel_y_i.to(dtype=TORCH_FLOAT),
                          x_mean = x_mean_element.to(dtype=TORCH_FLOAT), 
                          x_std = x_std_element.to(dtype=TORCH_FLOAT),
                          node_weight = nw.to(dtype=TORCH_FLOAT), 
                          L = lengthscale_element.to(dtype=TORCH_FLOAT),
                          pos = pos_i.to(dtype=TORCH_FLOAT),
                          pos_norm = (pos_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                          edge_index = edge_index, 
                          central_element_mask = central_element_mask, 
                          eid = torch.tensor(i))

        data_temp = data_temp.to(device_for_loading)

        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    t_load = time.time() - t_load 
    print('\ttook %g sec' %(t_load))
    return data_train_list, data_valid_list


def get_pygeom_dataset_lo_hi_pymech(data_xlo_path: str,
                       data_xhi_path: str,
                       edge_index_path_lo: str,
                       edge_index_path_hi: str,
                       node_weight: Optional[float] = 1.0,
                       device_for_loading : Optional[str] = 'cpu',
                       fraction_valid : Optional[float] = 0.1,
                       n_element_neighbors : Optional[int] = 0) -> Tuple[List,List]:
    t_load = time.time()
    print('In get_pygeom_dataset_lo_hi_pymech. Loading data and making pygeom dataset...')
    edge_index_lo = np.loadtxt(edge_index_path_lo, dtype=np.int64).T
    edge_index_lo = torch.tensor(edge_index_lo)
    edge_index_hi = np.loadtxt(edge_index_path_hi, dtype=np.int64).T
    edge_index_hi = torch.tensor(edge_index_hi)

    edge_index = edge_index_lo
    n_nodes_per_element = edge_index.max() + 1
    if n_element_neighbors > 0:
        node_max_per_element = edge_index.max()
        n_edges_per_element = edge_index.shape[1]
        edge_index_full = torch.zeros((2, n_edges_per_element*(n_element_neighbors+1)), dtype=edge_index.dtype)
        edge_index_full[:, :n_edges_per_element] = edge_index
        for i in range(1,n_element_neighbors+1):
            start = n_edges_per_element*i
            end = n_edges_per_element*(i+1)
            edge_index_full[:, start:end] = edge_index + (node_max_per_element+1)*i
        edge_index = edge_index_full

    # Load data
    xlo_field = readnek(data_xlo_path)
    xhi_field = readnek(data_xhi_path)

    # Train / valid split
    n_snaps = len(xlo_field.elem)

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

    # Get the element neighborhoods
    if n_element_neighbors > 0:
        Nelements = len(xlo_field.elem)
        pos_c = torch.zeros((Nelements, 3))
        for i in range(Nelements):
            pos_c[i] = torch.tensor(xlo_field.elem[i].centroid)
        edge_index_c = tgnn.knn_graph(x = pos_c, k = n_element_neighbors)

    # Get the element masks
    central_element_mask = torch.concat(
            (torch.ones((n_nodes_per_element), dtype=torch.int64),
             torch.zeros((n_nodes_per_element * n_element_neighbors), dtype=torch.int64))
            )
    central_element_mask = central_element_mask.to(torch.bool)

    data_train_list = []
    data_valid_list = []
    for i in range(n_snaps):
        pos_xlo_i = torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        vel_xlo_i = torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
        pos_xhi_i = torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
        vel_xhi_i = torch.tensor(xhi_field.elem[i].vel).reshape((3, -1)).T

        x_gll = xhi_field.elem[i].pos[0,0,0,:]
        dx_min = x_gll[1] - x_gll[0]

        error_max = (pos_xlo_i.max(dim=0)[0] - pos_xhi_i.max(dim=0)[0]).max()
        error_min = (pos_xlo_i.min(dim=0)[0] - pos_xhi_i.min(dim=0)[0]).max()
        rel_error_max = torch.abs(error_max / dx_min)*100
        rel_error_min = torch.abs(error_min / dx_min)*100
    
        # Check positions 
        if (rel_error_max > 1e-2) or (rel_error_min > 1e-2):
            print(f"Relative error in positions exceeds 0.01% in element i={i}.")
            sys.exit()
        if (pos_xlo_i.max() == 0. and pos_xlo_i.min() == 0.): 
            print(f"Node positions are not stored in {data_xlo_path}.")
            sys.exit()
        if (pos_xhi_i.max() == 0. and pos_xhi_i.min() == 0.): 
            print(f"Node positions are not stored in {data_xhi_path}.")
            sys.exit()

        # get x_mean and x_std 
        x_mean_element_lo = torch.mean(vel_xlo_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1)
        x_std_element_lo = torch.std(vel_xlo_i, dim=0).unsqueeze(0).repeat(central_element_mask.shape[0], 1)
        x_mean_element_hi = torch.mean(vel_xlo_i, dim=0).unsqueeze(0).repeat(vel_xhi_i.shape[0], 1)
        x_std_element_hi = torch.std(vel_xlo_i, dim=0).unsqueeze(0).repeat(vel_xhi_i.shape[0], 1)

        # element lengthscale 
        lengthscale_element = torch.norm(pos_xlo_i.max(dim=0)[0] - pos_xlo_i.min(dim=0)[0], p=2)

        # node weight 
        nw = torch.ones((vel_xhi_i.shape[0], 1)) * node_weight

        # Get the element neighbors for the input  
        if n_element_neighbors > 0:
            send = edge_index_c[0,:]
            recv = edge_index_c[1,:]
            nbrs = send[recv == i]

            pos_x_full = [pos_xlo_i]
            vel_x_full = [vel_xlo_i]
            for j in nbrs:
                pos_x_full.append( torch.tensor(xlo_field.elem[j].pos).reshape((3, -1)).T )
                vel_x_full.append( torch.tensor(xlo_field.elem[j].vel).reshape((3, -1)).T )
            pos_x_full = torch.concat(pos_x_full)
            vel_x_full = torch.concat(vel_x_full)

            # reset pos 
            pos_xlo_i = pos_x_full
            vel_xlo_i = vel_x_full

        # create data 
        data_temp = Data( x = vel_xlo_i.to(dtype=TORCH_FLOAT),
                          y = vel_xhi_i.to(dtype=TORCH_FLOAT),
                          x_mean_lo = x_mean_element_lo.to(dtype=TORCH_FLOAT),
                          x_std_lo = x_std_element_lo.to(dtype=TORCH_FLOAT),
                          x_mean_hi = x_mean_element_hi.to(dtype=TORCH_FLOAT),
                          x_std_hi = x_std_element_hi.to(dtype=TORCH_FLOAT),
                          node_weight = nw.to(dtype=TORCH_FLOAT),
                          L = lengthscale_element.to(dtype=TORCH_FLOAT),
                          pos_norm_lo = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                          pos_norm_hi = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                          edge_index_lo = edge_index,
                          edge_index_hi = edge_index_hi,
                          central_element_mask = central_element_mask,
                          eid = torch.tensor(i))

        # for synchronizing across element boundaries  
        if n_element_neighbors > 0:
            batch = None
            edge_index_coin = get_edge_index_coincident(
                    batch, data_temp.pos_norm_lo, data_temp.edge_index_lo)
            degree = utils.degree(edge_index_coin[1,:], num_nodes = data_temp.pos_norm_lo.shape[0])
            degree += 1.
            data_temp.edge_index_coin = edge_index_coin 
            data_temp.degree = degree
        else:
            data_temp.edge_index_coin = None
            data_temp.degree = None

        data_temp = data_temp.to(device_for_loading)

        if idx_train_mask[i] == 1:
            data_train_list.append(data_temp)
        else:
            data_valid_list.append(data_temp)

    t_load = time.time() - t_load 
    print('\ttook %g sec' %(t_load))
    return data_train_list, data_valid_list                           
    


def get_pygeom_dataset(data_x_path: str,
                       data_y_path: str,
                       edge_index_path: str,
                       node_element_ids_path: str,
                       global_ids_path: str,
                       pos_path: str, 
                       edge_index_vertex_path: Optional[str] = None,
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

    # Add extra edges if defined -- produces multiscale graph (adding P1 connectivity)
    if edge_index_vertex_path: 
        print('Adding p1 connectivity...')
        print('\tEdge index shape before: ', edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=NP_INT).T 
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        print('\tEdge index shape after: ', edge_index.shape)

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


