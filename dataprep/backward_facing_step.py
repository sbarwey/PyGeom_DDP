"""
Prepares PyGeom data from BFS VTK files obtained from foamToVTK
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import os,time,sys 
import numpy as np
import pyvista as pv 

import torch 
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn
import torch_geometric.transforms as transforms


def get_data_statistics(
        path_to_vtk : str, 
        multiple_cases : Optional[bool] = False ) -> List[np.ndarray]:
    #print('Reading vtk: %s' %(path_to_vtk))
    mesh = pv.read(path_to_vtk)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #print('Extracting data from vtk...')
    if multiple_cases:
        #print('\tmultiple cases...')
        data_full_temp = []
        time_vec = []

        # Get the case file list 
        case_path_list = mesh.field_data['case_path_list']
        n_cases = len(case_path_list)
        for c in range(n_cases):
            data_full_c = np.array(mesh.cell_data['x_%d' %(c)]) # [N_nodes x (N_features x N_snaps)]
            time_vec_c = np.array(mesh.field_data['time_%d' %(c)])
            field_names = np.array(mesh.field_data['field_list'])
            n_cells = mesh.n_cells
            n_features = len(field_names)
            n_snaps = len(time_vec_c)
            data_full_c = np.reshape(data_full_c, (n_cells, n_features, n_snaps), order='F')
            data_full_temp.append(data_full_c)
            time_vec.append(time_vec_c)
            
        # Concatenate data_full_temp and time_vec 
        data_full_temp = np.concatenate(data_full_temp, axis=2)
        time_vec = np.concatenate(time_vec)
        n_snaps = len(time_vec)
    else:
        #print('\tsingle case...')
        # Node features 
        data_full_temp = np.array(mesh.cell_data['x']) # [N_nodes x (N_features x N_snaps)]
        field_names = np.array(mesh.field_data['field_list'])
        time_vec = np.array(mesh.field_data['time'])
        n_cells = mesh.n_cells
        n_features = len(field_names)
        n_snaps = len(time_vec)
        data_full_temp = np.reshape(data_full_temp, (n_cells, n_features, n_snaps), order='F')

    # Do a dumb reshape
    data_full = np.zeros((n_snaps, n_cells, n_features), dtype=np.float32)
    for i in range(n_snaps):
        data_full[i,:,:] = data_full_temp[:,:,i]

    # Compute mean: 
    data_mean = data_full.mean(axis=(0,1), keepdims=False)
    data_std = data_full.std(axis=(0,1), keepdims=False)

    return [data_mean, data_std]



def get_pygeom_dataset_cell_data(
        path_to_vtk : str, 
        path_to_ei : str, 
        path_to_ea : str, 
        path_to_pos : str, 
        device_for_loading : str,
        use_radius : bool,
        time_skip : Optional[int] = 1,
        time_lag : Optional[int] = 1,
        scaling : Optional[list] = None,
        features_to_keep : Optional[list] = None, 
        fraction_valid : Optional[float] = 0.1, 
        multiple_cases : Optional[bool] = False ) -> tuple[list,list]:
    #print('Reading vtk: %s' %(path_to_vtk))
    mesh = pv.read(path_to_vtk)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #print('Extracting data from vtk...')
    if multiple_cases:
        #print('\tmultiple cases...')
        data_full_temp = []
        time_vec = []

        # Get the case file list 
        case_path_list = mesh.field_data['case_path_list']
        n_cases = len(case_path_list)
        for c in range(n_cases):
            data_full_c = np.array(mesh.cell_data['x_%d' %(c)]) # [N_nodes x (N_features x N_snaps)]
            time_vec_c = np.array(mesh.field_data['time_%d' %(c)])
            field_names = np.array(mesh.field_data['field_list'])
            n_cells = mesh.n_cells
            n_features = len(field_names)
            n_snaps = len(time_vec_c)
            data_full_c = np.reshape(data_full_c, (n_cells, n_features, n_snaps), order='F')
            data_full_temp.append(data_full_c)
            time_vec.append(time_vec_c)
            
        # Concatenate data_full_temp and time_vec 
        data_full_temp = np.concatenate(data_full_temp, axis=2)
        time_vec = np.concatenate(time_vec)
        n_snaps = len(time_vec)
    else:
        #print('\tsingle case...')
        # Node features 
        data_full_temp = np.array(mesh.cell_data['x']) # [N_nodes x (N_features x N_snaps)]
        field_names = np.array(mesh.field_data['field_list'])
        time_vec = np.array(mesh.field_data['time'])
        n_cells = mesh.n_cells
        n_features = len(field_names)
        n_snaps = len(time_vec)
        data_full_temp = np.reshape(data_full_temp, (n_cells, n_features, n_snaps), order='F')

    # Timestep reduction 
    # data_full_temp, time_vec, n_snaps 
    data_full_temp = data_full_temp[:, :, ::time_skip]
    time_vec = time_vec[::time_skip]
    n_snaps = len(time_vec)
    dt = time_vec[1:] - time_vec[:-1]

    # Do a dumb reshape
    data_full = np.zeros((n_snaps, n_cells, n_features), dtype=np.float32)
    for i in range(n_snaps):
        data_full[i,:,:] = data_full_temp[:,:,i]

    # Edge attributes and index, and node positions 
    #print('Reading edge index and node positions...')
    edge_index = torch.tensor(np.loadtxt(path_to_ei, dtype=np.longlong).T)
    #edge_attr = np.loadtxt(path_to_ea, dtype=np.float32)
    pos = torch.tensor(np.loadtxt(path_to_pos, dtype=np.float32))
  
    # Distance field
    #distance = np.array(mesh.cell_data['distance'], dtype=np.float32)
    #distance = np.reshape(distance, (n_cells, 1), order='F')
    distance = np.zeros((n_cells, 1))

    if use_radius:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create radius graph
        # -- outputs are edge_index and edge_attr
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        pos = torch.tensor(pos)
        radius = 0.001 # m
        self_loops = False
        max_num_neighbors = 30 
        edge_index_rad = tgnn.radius_graph(pos, r=radius, max_num_neighbors=max_num_neighbors)
        edge_index = torch.concat((edge_index_rad, torch.tensor(edge_index)), axis=1)

    # ~~~~ Populate edge_attr and coalesce edge_index
    data_ref = Data( pos = pos, edge_index = edge_index )
    cart = transforms.Cartesian(norm=False, max_value = None, cat = False)
    dist = transforms.Distance(norm = False, max_value = None, cat = True)

    # populate edge_attr
    cart(data_ref) # adds cartesian/component-wise distance
    dist(data_ref) # adds euclidean distance

    # extract edge_attr
    edge_attr = data_ref.edge_attr

    # Eliminate duplicate edges
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # change back to numpy 
    pos = np.array(pos)
    edge_attr = np.array(edge_attr)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create Time Lagged Data
    # if time_lag = 0, just copy data such that data_x = data_y 
    # if time_lag > 0, dataset size decreases by value of time_lag.
    #       -- data_y contains future snapshots for input data_x
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_snaps = data_full.shape[0] - time_lag
    data_x = [] 
    data_y = []
    time_y = []
    for i in range(n_snaps): 
        data_x.append([data_full[i]]) 
        if time_lag == 0:
            y_temp = [data_full[i]]
            time_temp = [time_vec[i]]
        else:
            y_temp = []
            time_temp = [] 
            for t in range(1, time_lag+1):
                y_temp.append(data_full[i+t])
                time_temp.append(time_vec[i+t])
        data_y.append(y_temp)
        time_y.append(time_temp)
    
    data_x = np.array(data_x) # shape: [n_snaps, 1, n_nodes, n_features]
    data_y = np.array(data_y) # shape: [n_snaps, time_lag, n_nodes, n_features]
    time_y = np.array(time_y) # shape: [n_snaps, time_lag] 

    if time_lag > 0:
        time_vec = time_vec[:-time_lag] 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shuffle -- train/valid split
    # set aside 10% of the data for validation  
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #print('arranging data for train/valid split...')
    #print('\tvalidation size is %g * n_data' %(fraction_valid))
    
    if fraction_valid > 0:
        # How many total snapshots to extract 
        n_full = n_snaps
        n_valid = int(np.floor(fraction_valid * n_full))

        # Get validation set indices 
        idx_valid = np.sort(np.random.choice(n_full, n_valid, replace=False))

        # Get training set indices 
        idx_train = np.array(list(set(list(range(n_full))) - set(list(idx_valid))))

        # Train/test split 
        data_x_train = data_x[idx_train]
        data_y_train = data_y[idx_train]

        data_x_valid = data_x[idx_valid]
        data_y_valid = data_y[idx_valid]

        time_vec_train = time_vec[idx_train]
        time_vec_valid = time_vec[idx_valid]

        time_y_train = time_y[idx_train]
        time_y_valid = time_y[idx_valid]

        n_train = n_full - n_valid
    else:
        data_x_train = data_x
        data_y_train = data_y
        data_x_valid = None
        data_y_valid = None
        time_vec_train = time_vec
        time_vec_valid = time_vec
        time_y_train = time_y
        time_y_valid = time_y
        n_full = n_snaps
        n_valid = 0
        n_train = n_full
    
    #print('\tn_full: ', n_full)
    #print('\tn_train: ', n_train)
    #print('\tn_valid: ', n_valid)

    #print('data_x_train: ', data_x_train.shape)
    #print('data_y_train: ', data_y_train.shape)
    #print('data_x_valid: ', data_x_valid.shape)
    #print('data_y_valid: ', data_y_valid.shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Scaling node features
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    eps = 1e-10
    if scaling:
        data_train_mean = scaling[0]
        data_train_std = scaling[1]
    else:
        data_train_mean = np.zeros(n_features)
        data_train_std = np.ones(n_features)

    data_train_mean = np.reshape(data_train_mean, (1,1,1,-1))
    data_train_std = np.reshape(data_train_std, (1,1,1,-1))

    data_x_train = (data_x_train - data_train_mean)/(data_train_std + eps)
    data_y_train = (data_y_train - data_train_mean)/(data_train_std + eps)
    if fraction_valid > 0:
        data_x_valid = (data_x_valid - data_train_mean)/(data_train_std + eps)
        data_y_valid = (data_y_valid - data_train_mean)/(data_train_std + eps)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize edge attributes 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    edge_attr_mean = edge_attr.mean()
    edge_attr_std = edge_attr.std()
    edge_attr = (edge_attr - edge_attr_mean)/(edge_attr_std + eps) 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make pyGeom dataset 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    data_x_train = torch.tensor(data_x_train, dtype=torch.float32)
    data_y_train = torch.tensor(data_y_train, dtype=torch.float32)
    time_vec_train = torch.tensor(time_vec_train)
    time_y_train = torch.tensor(time_y_train)
    if fraction_valid > 0:
        data_x_valid = torch.tensor(data_x_valid, dtype=torch.float32)
        data_y_valid = torch.tensor(data_y_valid, dtype=torch.float32)
        time_vec_valid = torch.tensor(time_vec_valid)
        time_y_valid = torch.tensor(time_y_valid)
    
    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    pos = torch.tensor(pos)
    distance = torch.tensor(distance)
    
    # Re-sort:  
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # Add self loops:
    #edge_index, edge_attr = utils.add_self_loops(edge_index, edge_attr, fill_value='mean')

    # Restrict data based on features_to_keep: 
    if features_to_keep == None:
        n_features = data_x_train[0].shape[-1]
        features_to_keep = list(range(n_features))
    data_x_train = data_x_train[:,:,:,features_to_keep]
    data_y_train = data_y_train[:,:,:,features_to_keep]
    if fraction_valid > 0:
        data_x_valid = data_x_valid[:,:,:,features_to_keep]
        data_y_valid = data_y_valid[:,:,:,features_to_keep]
    data_train_mean = data_train_mean[:,:,:,features_to_keep]
    data_train_std = data_train_std[:,:,:,features_to_keep]

    data_train_mean = torch.tensor(data_train_mean)
    data_train_std = torch.tensor(data_train_std)

    # Training 
    data_train_list = []
    for i in range(n_train):
        if time_lag == 0:
            y_temp = data_y_train[i,0]
        else:
            y_temp = [] 
            for t in range(time_lag):
                y_temp.append(data_y_train[i,t])

        data_temp = Data(   x = data_x_train[i,0],
                            y = y_temp,
                            distance = distance, 
                            edge_index = edge_index,
                            edge_attr = edge_attr,
                            pos = pos,
                            bounding_box = [pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()],
                            data_scale = (data_train_mean, data_train_std),
                            edge_scale = (edge_attr_mean, edge_attr_std),
                            t_x = time_vec_train[i],
                            t_y = time_y_train[i],
                            field_names = field_names)
        data_temp = data_temp.to(device_for_loading)
        data_train_list.append(data_temp)

    # Testing: 
    data_valid_list = []
    if fraction_valid > 0: 
        for i in range(n_valid):
            if time_lag == 0:
                y_temp = data_y_valid[i,0]
            else:
                y_temp = [] 
                for t in range(time_lag):
                    y_temp.append(data_y_valid[i,t])

            data_temp = Data(   x = data_x_valid[i,0],
                                y = y_temp,
                                distance = distance,
                                edge_index = edge_index,
                                edge_attr = edge_attr,
                                pos = pos,
                                bounding_box = [pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()],
                                data_scale = (data_train_mean, data_train_std),
                                edge_scale = (edge_attr_mean, edge_attr_std),
                                t_x = time_vec_valid[i],
                                t_y = time_y_valid[i],
                                field_names = field_names)
            data_temp = data_temp.to(device_for_loading)
            data_valid_list.append(data_temp)

    # print('\n\tTraining samples: ', len(data_train_list))
    # print('\tValidation samples: ', len(data_valid_list))
    # print('\tN_nodes: ', data_train_list[0].x.shape[0])
    # print('\tN_edges: ', data_train_list[0].edge_index.shape[1])
    # print('\tN_features: ', data_train_list[0].x.shape[1])
    # print('\thas_self_loops: ', data_train_list[0].has_self_loops())
    # print('\n')

    return data_train_list, data_valid_list
