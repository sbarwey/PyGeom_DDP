"""
Postprocess trained model (no DDP) 
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging
import copy 

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
import models.gnn as gnn 

# Data preparation
import dataprep.speedy as spd

torch.manual_seed(122)
np.random.seed(122)
torch.set_grad_enabled(False)
SMALL = 1e-10

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Postprocess training losses: ORIGINAL 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0: 
    a = torch.load('./saved_models/NO_NOISE_GNN_ROLLOUT_1_SEED_122_4_3_128_4_2_2_2_1_16.tar')
    fig, ax = plt.subplots()
    ax.plot(a['loss_hist_train'][1:], lw=2)
    ax.plot(a['loss_hist_test'], lw=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.show(block=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load model, and evaluate on training set 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    path_to_data = "./datasets/speedy_numpy_file_train.npz"
    device_for_loading = "cpu"
    rollout_length = 10
    print('Loading dataset...')
    data_list, _ = spd.get_pygeom_dataset(
            path_to_data = path_to_data,
            device_for_loading = device_for_loading,
            time_lag = rollout_length, 
            fraction_valid = 0.0)
    print('Done loading dataset.')

    # # Get data: 
    # sample = data_list[0]
    # field_id = 3 
    # f_plt = sample.x[:,field_id].reshape(sample.field_shape) 
    # fig, ax = plt.subplots()
    # ax.imshow(f_plt)
    # plt.show(block=False)

    # Load model 
    device = 'cpu'
    modelpath_topk = './saved_models/NO_NOISE_GNN_ROLLOUT_1_SEED_122_4_3_128_4_2_2_2_1_16.tar'
    p = torch.load(modelpath_topk)
    input_dict = p['input_dict']
    if 'n_mmp_layers' not in input_dict:
        input_dict['n_mmp_layers'] = 1
    model_topk = gnn.TopkMultiscaleGNN(
            input_dict['input_node_channels'],
            input_dict['input_edge_channels'],
            input_dict['hidden_channels'],
            input_dict['output_node_channels'],
            input_dict['n_mlp_hidden_layers'],
            input_dict['n_mmp_layers'],
            input_dict['n_messagePassing_layers'],
            input_dict['max_level_mmp'],
            input_dict['l_char'],
            input_dict['max_level_topk'],
            input_dict['rf_topk'],
            input_dict['name']) 
    model_topk.load_state_dict(p['state_dict'])
    model_topk.to(device)
    model_topk.eval()

    # ~~~~~~~~~~~~~~~~~~~~
    # Predictions 
    i = 6000
    data = data_list[i]
    
    n_nodes = data.x.shape[0]
    n_features = data.x.shape[1]
    traj_input_ss = np.zeros((rollout_length, n_nodes, n_features)) 
    traj_output_ss = np.zeros_like(traj_input_ss)
    traj_error_ss = np.zeros_like(traj_input_ss)
    traj_mask_ss = np.zeros((rollout_length, n_nodes)) 
    traj_target = np.zeros_like(traj_input_ss)

    # Rollout prediction: 
    data_mean = data.data_mean
    data_std = data.data_std
    x_new = (data.x - data_mean)/(data_std + SMALL) # scaled input
    for t in range(rollout_length):
        print(f"Iter {t+1}")
        # ~~~~ Rollout predictions ~~~~ #
        x_old = torch.clone(x_new)
        x_src, mask = model_topk(x_old, data.edge_index, data.pos, data.edge_attr, data.batch)
        x_new = x_old + x_src
        # ~~~~ Single-step predictions ~~~~ #
        if t == 0:
            x_old_ss = x_old
        else:
            x_old_ss = (data.y[t-1] - data_mean)/(data_std + SMALL)
        x_src, mask_ss = model_topk(x_old_ss, data.edge_index, data.pos, data.edge_attr, data.batch)
        x_new_ss = x_old_ss + x_src
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

        # QoIs 
        x_old_unscaled = x_old*(data_std + SMALL) + data_mean
        x_new_unscaled = x_new*(data_std + SMALL) + data_mean
        x_old_ss_unscaled = x_old_ss*(data_std + SMALL) + data_mean
        x_new_ss_unscaled = x_new_ss*(data_std + SMALL) + data_mean
        target = data.y[t]

        # store 
        traj_input_ss[t] = x_old_ss_unscaled
        traj_output_ss[t] = x_new_ss_unscaled
        traj_error_ss[t] = x_new_ss_unscaled - target
        traj_mask_ss[t] = mask_ss
        traj_target[t] = target



    # Plot error 
    if 1 == 1:
        sid = 0
        fig, ax = plt.subplots(1,4,figsize=(15,4), sharex=True)
        for fid in range(n_features):
            ax[fid].hist((traj_error_ss[sid, :, fid])/(data_std[0,fid] + SMALL), bins=100)
            ax[fid].set_ylabel('Freq')
            ax[fid].set_xlabel('Error -- field %d' %(fid))
            ax[fid].set_ylim([0, 400])
        plt.show(block=False)

        pass 

    # Plot predictions 
    if 1 == 0:
        sample = data_list[0]
        for sid in range(rollout_length):
            print(f"Plotting step {sid}")
            fig, ax = plt.subplots(3,4,figsize=(15,7),gridspec_kw = {'wspace':0.01, 'hspace':0.01})
            for fid in range(n_features):
                f_plt = traj_input_ss[sid,:,fid].reshape(sample.field_shape)
                ax[0,fid].imshow(f_plt)
                #ax[0,fid].set_aspect('equal')
                ax[0,fid].grid(False)
                ax[0,fid].set_xticklabels([])
                ax[0,fid].set_yticklabels([])
                ax[0,fid].set_title(f"Step {sid}")
                ax[0,0].set_ylabel('Input')

                f_plt = traj_mask_ss[sid,:].reshape(sample.field_shape)
                ax[1,fid].imshow(f_plt)
                #ax[1,fid].set_aspect('equal')
                ax[1,fid].grid(False)
                ax[1,fid].set_xticklabels([])
                ax[1,fid].set_yticklabels([])
                ax[1,0].set_ylabel('Mask')

                f_plt = traj_output_ss[sid,:,fid].reshape(sample.field_shape)
                ax[2,fid].imshow(f_plt)
                #ax[2,fid].set_aspect('equal')
                ax[2,fid].grid(False)
                ax[2,fid].set_xticklabels([])
                ax[2,fid].set_yticklabels([])
                ax[2,0].set_ylabel('Forecast (SS)')
            plt.savefig("./outputs/movies/step_%.4d.png" %(sid), dpi=600)
            plt.close()
            #plt.show(block=False)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read in raw data 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    a = np.load("./datasets/speedy_numpy_file.npz")
    lst = a.files # ['temperature_950', 'specific_humidity_950', 'u_wind_500', 'v_wind_500']
    data_1 = a['temperature_950']
    data_2 = a['specific_humidity_950']
    data_3 = a['u_wind_500']
    data_4 = a['v_wind_500']
    
    idx = 0
    img = data_1[idx]
    img_shape = img.shape
    x_lim = np.linspace(0,1,img_shape[0])
    y_lim = np.linspace(0,1,img_shape[1])

    # get pos 
    X,Y = np.meshgrid(x_lim, y_lim, indexing='ij')
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    pos = torch.cat( (X.reshape(-1,1), Y.reshape(-1,1)), axis=1 )
    pos_rot = torch.cat( (Y.reshape(-1,1), -X.reshape(-1,1) + 1), axis=1 )

    i = 3 
    data = data_4
    fig, ax = plt.subplots(1,4)
    ax[0].pcolormesh(Y, X, data[i])
    ax[1].imshow(data[i])
    ax[2].scatter(pos[:,0], pos[:,1], c=data[i].reshape((-1,1)))
    ax[3].scatter(pos_rot[:,0], pos_rot[:,1], c=data[i].reshape((-1,1)))

    ax[0].set_title(f"{i}")
    ax[1].set_title(f"{i}")
    ax[2].set_title(f"{i}")
    ax[3].set_title(f"{i}")
    plt.show(block=False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Partition data 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    path_to_full_data = "./datasets/speedy_numpy_file.npz"
    a = np.load(path_to_full_data)
    field_names = a.files
    N_snaps = a[field_names[0]].shape[0]

    # # Split into train/test
    # split = 0.5 
    # idx_split = int(N_snaps * split)
    # 
    # path_to_train_data = "./datasets/speedy_numpy_file_train.npz"
    # path_to_test_data = "./datasets/speedy_numpy_file_test.npz"

    # # np.savez(outfile, x=x, y=y) 

    # fields_train = []
    # fields_test = []
    # for f in field_names:
    #     field_data = a[f]
    #     fields_train.append(field_data[:idx_split])
    #     fields_test.append(field_data[idx_split:])

    # np.savez(path_to_train_data, 
    #          temperature_950 = fields_train[0],
    #          specific_humidity_950 = fields_train[1],
    #          u_wind_500 = fields_train[2],
    #          v_wind_500 = fields_train[3])

    # np.savez(path_to_test_data, 
    #          temperature_950 = fields_test[0],
    #          specific_humidity_950 = fields_test[1],
    #          u_wind_500 = fields_test[2],
    #          v_wind_500 = fields_test[3])

    # Read 
    path_to_train_data = "./datasets/speedy_numpy_file_train.npz"
    a = np.load(path_to_train_data)
    field_names = a.files
    N_snaps = a[field_names[0]].shape[0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test PyGeom data 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 1:
    path_to_data = "./datasets/speedy_numpy_file_train.npz"
    device_for_loading = "cpu"
    data_train_list, data_valid_list = spd.get_pygeom_dataset(
            path_to_data = path_to_data,
            device_for_loading = device_for_loading,
            time_lag = 1, 
            fraction_valid = 0.1)
    
    # Get data: 
    sample = data_train_list[0]
    # field_id = 3 
    # f_plt = sample.x[:,field_id].reshape(sample.field_shape) 
    # fig, ax = plt.subplots()
    # ax.imshow(f_plt)
    # plt.show(block=False)

    # Test model 
    # for baseline -- n_mmp_layers = 2, max_level_topk = 0 
    # for finetune -- n_mmp_layers = 1, max_level_topk = 1


    # Baseline model 
    input_node_channels = sample.x.shape[1]
    input_edge_channels = sample.edge_attr.shape[1]
    hidden_channels = 32
    output_node_channels = input_node_channels
    n_mlp_hidden_layers = 2 
    n_mmp_layers = 2 
    n_messagePassing_layers = 2 
    max_level_mmp = 2
    l_char = 1
    max_level_topk = 0
    rf_topk = 4

    model_bl = gnn.TopkMultiscaleGNN(
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

    input_dict_bl = model_bl.input_dict()
    
    # Topk model 
    model = gnn.TopkMultiscaleGNN(
            input_node_channels = input_dict_bl['input_node_channels'],
            input_edge_channels = input_dict_bl['input_edge_channels'],
            hidden_channels = input_dict_bl['hidden_channels'],
            output_node_channels = input_dict_bl['output_node_channels'],
            n_mlp_hidden_layers = input_dict_bl['n_mlp_hidden_layers'],
            n_mmp_layers = 1,
            n_messagePassing_layers = input_dict_bl['n_messagePassing_layers'],
            max_level_mmp = input_dict_bl['max_level_mmp'],
            l_char = input_dict_bl['l_char'],
            max_level_topk = 1,
            rf_topk = 4,
            name='gnn')


    # weight.detach().clone()
    # bias.detach().clone()
    # mmp_layer.copy(mmp_layer) 
    # model.set_mmp_layer(model_bl.mmp_down[0][1], 

    w_bl = model_bl.mmp_down[0][1].processors_up[0][0].edge_updater.mlp[1].weight
    w = model.mmp_up[0][0].processors_up[0][0].edge_updater.mlp[1].weight
    print(f"w_bl_before: {w_bl}")
    print(f"w_before: {w}")

    print('params before:', count_parameters(model))
    # copy mmp layers  
    model.mmp_down[0][0].copy(model_bl.mmp_down[0][0], freeze_params=True)
    model.mmp_up[0][0].copy(model_bl.mmp_down[0][1], freeze_params=True)
    # copy node encoder 
    model.node_encoder.copy(model_bl.node_encoder, freeze_params=True)
    # copy edge encoder 
    model.edge_encoder.copy(model_bl.edge_encoder, freeze_params=True)
    # copy node decoder 
    model.node_decoder.copy(model_bl.node_decoder, freeze_params=True)
    print('params after:', count_parameters(model))

    w_bl = model_bl.mmp_down[0][1].processors_up[0][0].edge_updater.mlp[1].weight
    w = model.mmp_up[0][0].processors_up[0][0].edge_updater.mlp[1].weight
    print(f"w_bl_after: {w_bl}")
    print(f"w_after: {w}")


    # # processors_down 
    # for i in range(mmp_layer.n_levels):
    #     for j in range(model.n_messagePassing_layers):
    #         print(i,j)
    #         # ~~~~ Enter processor layer copy 
    #         mp_layer = mmp_layer.processors_down[i][j]
    #         mp_layer_bl = mmp_layer_bl.processors_down[i][j]
    #         
    #         # ~~~~ Enter MP layer copy 
    #         # 1) edge updater MLP 
    #         mlp = mp_layer.edge_updater
    #         mlp_bl = mp_layer_bl.edge_updater 

    #         # ~~~~ Enter MLP copy 
    #         mlp.norm_layer.weight[:] = mlp_bl.norm_layer.weight.detach().clone()
    #         mlp.norm_layer.bias[:] = mlp_bl.norm_layer.bias.detach().clone() 
    #         for k in range(len(mlp.mlp)):
    #             mlp.mlp[k].weight[:,:] = mlp_bl.mlp[k].weight.detach().clone()
    #             mlp.mlp[k].bias[:] = mlp_bl.mlp[k].bias.detach().clone()

    #         asdf
    #         pass



    # # Forward pass 
    # with torch.no_grad(): 
    #     
    #     # Scale input  
    #     eps = 1e-10
    #     x_scaled = (sample.x - sample.data_mean)/(sample.data_std + eps)
    #     out_gnn = model(x_scaled, sample.edge_index, sample.pos, sample.edge_attr)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot graphs at different lengthscales 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    import torch_geometric.nn as tgnn
    from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x
    path_to_data = "./datasets/speedy_numpy_file_train.npz"
    device_for_loading = "cpu"
    data_train_list, data_valid_list = spd.get_pygeom_dataset(
            path_to_data = path_to_data,
            device_for_loading = device_for_loading,
            time_lag = 3, 
            fraction_valid = 0.1)

    
    # Get edge attr 
    edge_attr = data_train_list[0].edge_attr
    mean_edge_length = edge_attr[:,2].mean()
    l_char = mean_edge_length 


    # First 1 
    x_fine = data_train_list[0].x
    ei_fine = data_train_list[0].edge_index 
    ea_fine = data_train_list[0].edge_attr
    pos_fine = data_train_list[0].pos 
    batch_fine = ei_fine.new_zeros(x_fine.size(0))
    
    fig, ax = plt.subplots(figsize=(8,7))
    ax.scatter(pos_fine[:,0], pos_fine[:,1], color='gray')
    edge_xyz = pos_fine[ei_fine].permute(1,0,2)
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="black", lw=0.1, alpha=0.3)

    cluster = tgnn.voxel_grid(
                    pos = pos_fine,
                    size = l_char * 2,
                    batch = batch_fine)

    x_crse, ei_crse, ea_crse, batch_crse, pos_crse, _, _ = avg_pool_mod(
                    cluster,
                    x_fine,  
                    ei_fine,
                    ea_fine,
                    batch_fine,
                    pos_fine)

    # Plot 
    ax.scatter(pos_crse[:,0], pos_crse[:,1], color='red')
    edge_xyz = pos_crse[ei_crse].permute(1,0,2)
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="red", lw=1, alpha=0.3)

    # second 
    x_fine = x_crse
    ei_fine = ei_crse
    ea_fine = ea_crse
    pos_fine = pos_crse
    batch_fine = batch_crse

    cluster = tgnn.voxel_grid(
                    pos = pos_fine,
                    size = l_char * 2 * 2,
                    batch = batch_fine)

    x_crse, ei_crse, ea_crse, batch_crse, pos_crse, _, _ = avg_pool_mod(
                    cluster,
                    x_fine,  
                    ei_fine,
                    ea_fine,
                    batch_fine,
                    pos_fine)

    ax.scatter(pos_crse[:,0], pos_crse[:,1], color='blue')
    edge_xyz = pos_crse[ei_crse].permute(1,0,2)
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="blue", lw=1, alpha=0.3)

    ax.grid(False)
    plt.show(block=False)


