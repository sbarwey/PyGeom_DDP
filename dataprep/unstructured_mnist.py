"""
Prepares PyGeom dataset from MNIST VTK files. 
    - extract 5s and 6s only  
"""
from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable
import numpy as np
import pyvista as pv 
import torch 
from torch_geometric.data import Data
import torch_geometric.utils as utils 


def get_mnist_dataset(path_to_vtk: str, 
                      path_to_ei: str, 
                      path_to_ea: str, 
                      path_to_pos: str, 
                      device_for_loading: str) -> tuple[list,list]:

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load VTK 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mesh = pv.read(path_to_vtk)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~ Load full datasets 
    mnist_train = np.array(mesh.cell_data['mnist_train'])
    mnist_test = np.array(mesh.cell_data['mnist_test'])
    labels_train = np.array(mesh.field_data['labels_train'])
    labels_test = np.array(mesh.field_data['labels_test'])

    # ~~~~ Standardize Images 
    min_val = mnist_train.min() # 0.0
    max_val = mnist_train.max() # 255.0
    mnist_train = (mnist_train - min_val)/(max_val - min_val) 
    mnist_test = (mnist_test - min_val)/(max_val - min_val) 

    # ~~~~ Create smaller dataset by extracting only 5s and 6s: 
    # ~~~~ Training:
    idx_5 = labels_train == 5
    idx_6 = labels_train == 6

    # Isolate subset 
    mnist_train_5 = mnist_train[:,idx_5][:,:500]
    mnist_train_6 = mnist_train[:,idx_6][:,:500] 
    labels_train_5 = labels_train[idx_5][:500]
    labels_train_6 = labels_train[idx_6][:500]

    # # Concatenate: 
    # mnist_train_reduced = np.concatenate((mnist_train_5, mnist_train_6), axis=1)
    # labels_train_reduced = np.concatenate((labels_train_5, labels_train_6))
    # labels_train_reduced[labels_train_reduced == 5] = 0 
    # labels_train_reduced[labels_train_reduced == 6] = 1 

    # ~~~~ Testing: 
    idx_5 = labels_test == 5
    idx_6 = labels_test == 6 

    # Isolate subset 
    mnist_test_5 = mnist_test[:,idx_5][:,:100]
    mnist_test_6 = mnist_test[:,idx_6][:,:100]
    labels_test_5 = labels_test[idx_5][:100]
    labels_test_6 = labels_test[idx_6][:100]

    # # Concatenate:
    # mnist_test_reduced = np.concatenate((mnist_test_5, mnist_test_6), axis=1)
    # labels_test_reduced = np.concatenate((labels_test_5, labels_test_6))
    # labels_test_reduced[labels_test_reduced == 5] = 0
    # labels_test_reduced[labels_test_reduced == 6] = 1

    # ~~~~ Load edge index (adjacency) 
    edge_index = np.loadtxt(path_to_ei, dtype=np.long).T

    # ~~~~ Load edge attributes 
    edge_attr = np.loadtxt(path_to_ea, dtype=np.float32) 

    # ~~~~ Load node positions 
    pos = np.loadtxt(path_to_pos, dtype=np.float32)

    # ~~~~ Get number of cells and number of edges 
    N_cells = mnist_train_5.shape[0]
    N_edges = edge_index.shape[1]

    N_train = mnist_train_5.shape[1]
    N_test = mnist_test_5.shape[1]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make PyGeom dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert to torch tensor
    mnist_train_5 = torch.tensor(mnist_train_5)
    mnist_test_5 = torch.tensor(mnist_test_5)
    mnist_train_6 = torch.tensor(mnist_train_6)
    mnist_test_6 = torch.tensor(mnist_test_6)

    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)
    pos = torch.tensor(pos)

    # Re-sort edge index and edge attr  
    edge_index, edge_attr = utils.coalesce(edge_index, edge_attr)

    # Training 
    data_train_list = [] 
    for i in range(N_train):
        # print('Train %d/%d' %(i+1, N_train))
        data_temp = Data(   x = mnist_train_5[:,i].reshape(-1,1),
                            edge_index = edge_index, 
                            edge_attr = edge_attr, 
                            pos = pos,
                            y = mnist_train_6[:,i].reshape(-1,1))

        data_temp = data_temp.to(device_for_loading)

        data_train_list.append(data_temp)

    data_test_list = []
    for i in range(N_test):
        # print('Test %d/%d' %(i+1, N_test))
        data_temp = Data(   x = mnist_test_5[:,i].reshape(-1,1),
                            edge_index = edge_index, 
                            edge_attr = edge_attr, 
                            pos = pos,
                            y = mnist_test_6[:,i].reshape(-1,1))
        data_temp = data_temp.to(device_for_loading)
        data_test_list.append(data_temp)

    return data_train_list, data_test_list 




