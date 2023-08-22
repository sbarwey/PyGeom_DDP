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
import models.cnn as cnn 
import models.gnn as gnn 

# Data preparation
import dataprep.unstructured_mnist as umnist
import dataprep.backward_facing_step as bfs

def scalar2openfoam(image_vec, filename, objectname, time_value):
    time_write = time.time()
    with open(filename, 'w') as f:
        # Openfoam header:
        f.write('/*--------------------------------*- C++ -*----------------------------------*\\\n')
        f.write('| =========                 |                                                 |\n')
        f.write('| \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n')
        f.write('|  \\\    /   O peration     | Version:  v2006                                 |\n')
        f.write('|   \\\  /    A nd           | Web:      www.OpenFOAM.org                      |\n')
        f.write('|    \\\/     M anipulation  |                                                 |\n')
        f.write('\\*---------------------------------------------------------------------------*/\n')
        
        # FoamFile part:
        f.write('FoamFile\n')
        f.write('{\n')
        f.write('\tversion\t\t2.0;\n')
        f.write('\tformat\t\tascii;\n')
        f.write('\tclass\t\tvolScalarField;\n')
        f.write('\tlocation\t"%g";\n' %(time_value))
        f.write('\tobject\t\t%s;\n' %(objectname))
        f.write('}\n')
        f.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        f.write('\n')
        
        # internalField
        f.write('dimensions\t\t[0 0 0 0 0 0 0];\n\n')
        f.write('internalField\tnonuniform List<scalar>\n')
        f.write('%s\n' %(len(image_vec)))
        f.write('(\n')
        for val in image_vec:
            f.write('%s\n' %(val))
        f.write(')\n')
        f.write(';\n\n') 

        # boundaryField -- do cyclic / empty for now  
        f.write('boundaryField\n')
        f.write('{\n')
        
        # left:
        f.write('\tinlet\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tzeroGradient;\n')
        f.write('\t}\n')
        
        # right:
        f.write('\toutlet\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tfixedValue;\n')
        f.write('\t\tvalue\t\t\tuniform 0;\n')
        f.write('\t}\n')
        
        # top:
        f.write('\t\"(lowerWallStartup|upperWallStartup)\"\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tsymmetryPlane;\n')
        f.write('\t}\n')
        
        # bottom: 
        f.write('\t\"(upperWall|lowerWall)\"\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tzeroGradient;\n')
        f.write('\t}\n')

        # front:
        f.write('\t\"(front|back)\"\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tempty;\n')
        f.write('\t}\n')

        #oldInternalFaces
        f.write('\t\"oldInternalFaces\"\n')
        f.write('\t{\n')
        f.write('\t\ttype\t\t\tinternal;\n')
        f.write('\t\tvalue\t\t\tuniform 0;\n')
        f.write('\t}\n')
        f.write('}\n\n')

        f.write('// ************************************************************************* //')
    time_write = time.time() - time_write 
    print('%s: \t\t\t%.5e s' %(filename, time_write))

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

use_radius = True
test_dataset, _ = bfs.get_pygeom_dataset_cell_data(
                vtk_file_test, 
                path_to_ei, 
                path_to_ea,
                path_to_pos, 
                device_for_loading, 
                use_radius,
                time_lag = 2,
                scaling = [data_mean, data_std],
                features_to_keep = [1,2], 
                fraction_valid = 0, 
                multiple_cases = False)

# Remove first snapshot
test_dataset.pop(0)

seed_list = None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Postprocess training losses: ORIGINAL 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0: 
    print('Postprocess training losses (original)')
    # Load model 
    # #a = torch.load('saved_models/model_single_scale.tar')
    # a = torch.load('saved_models/model_multi_scale.tar')
    # #c = torch.load('saved_models/model_multi_scale_topk.tar.old')
    # #c = torch.load('saved_models/topk_down_topk_1_1_up_topk_1_1_factor_16_hc_128_down_enc_4_up_enc_down_dec_4_4_4_up_dec_4_4_param_sharing_0.tar')
    # b = torch.load('saved_models/topk_unet_down_topk_2_up_topk_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_1.tar')
    # c = torch.load('saved_models/topk_unet_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')

    # # Effect of MMP blocks and parameter sharing 
    # a = torch.load('saved_models/topk_unet_down_topk_1_up_topk_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # #b = torch.load('saved_models/topk_unet_down_topk_2_up_topk_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # b = torch.load('saved_models/topk_unet_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # c = torch.load('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # d = torch.load('saved_models/topk_unet_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')

    # # Effect of RK2
    # a = torch.load('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # b = torch.load('/Users/sbarwey/Files/ml/DDP_PyGeom_testing/saved_models/topk_unet_rollout_1_rk2_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')


    # Effect of transfer learning
    seed = 42
    a = torch.load('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    a_label = 'Baseline (rollout = 1)'

    b = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    b_label = 'Baseline + TopK (rollout = 1)'

    c = torch.load('saved_models/pretrained_topk_unet_rollout_2_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    c_label = 'Baseline + TopK (rollout = 2)'

    d = torch.load('saved_models/pretrained_topk_unet_rollout_3_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    d_label = 'Baseline + TopK (rollout = 3)'



    # Effect of seed: 
    a = torch.load('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    a_label = 'Baseline (rollout = 1)'

    b = torch.load('saved_models/NO_RADIUS_topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    b_label = 'Baseline, No Radius (rollout = 1)'

    # seed = 42
    # b = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    # b_label = 'TopK: Seed 1'
    # 
    # seed = 65
    # c = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    # c_label = 'TopK: Seed 2'

    # seed = 82
    # d = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    # d_label = 'TopK: Seed 3'

    # seed = 105
    # e = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
    # e_label = 'TopK: Seed 4'

    # Plot losses:
    fig, ax = plt.subplots(1,4,sharey=False, sharex=True, figsize=(14,6))
    ax[0].plot(a['loss_hist_train'])
    ax[0].plot(a['loss_hist_test'])
    ax[0].set_yscale('log')
    #ax[0].set_ylim([5e-4, 1e-3])
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    #ax[0].set_title('1x MMP (rollout = 2)')
    ax[0].set_title(a_label)

    ax[1].plot(b['loss_hist_train'], label='train')
    ax[1].plot(b['loss_hist_test'], label='valid')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Epochs')
    #ax[1].set_title('2x MMP (rollout = 2)')
    ax[1].set_title(b_label)

    ax[2].plot(c['loss_hist_train'])
    ax[2].plot(c['loss_hist_test'])
    ax[2].set_yscale('log')
    ax[2].set_xlim([0,300])
    ax[2].set_xlabel('Epochs')
    ax[2].set_title(c_label)

    ax[3].plot(d['loss_hist_train'])
    ax[3].plot(d['loss_hist_test'])
    ax[3].set_yscale('log')
    ax[3].set_xlim([0,200])
    ax[3].set_xlabel('Epochs')
    ax[3].set_title(d_label)

    plt.show(block=False)


    # Combined loss plot 
    baseline_loss = np.mean(a['loss_hist_train'][-10:])
    #combined = [b,c,d,e]
    combined = [b]

    fig, ax = plt.subplots()
    ax.axhline(y=baseline_loss, color='black', linestyle='--', lw=2)
    for i in range(len(combined)):
        ax.plot(combined[i]['loss_hist_train'][1:], lw=2)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Epochs')
    plt.show(block=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Overwrite some model names: 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    print('Overwrite some model names...')
    seed_list = [105, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 
                 42, 65, 82]

    for seed in seed_list:
        b = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
        model_name = b['input_dict']['name']
        if 'seed' not in model_name: 
            print('\tcorrecting name for model of seed ', seed)
            rollout_steps = 1
            model_name_new = 'pretrained_topk_unet_rollout_%d_seed_%d' %(rollout_steps, seed)
            b['input_dict']['name'] = model_name_new
            
            torch.save(b, 'saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Postprocess training losses: FOCUS ON EFFECT OF SEEDING 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0: 
    print('Postprocess training losses: focus on effect of seeding.')
    # baseline:
    a = torch.load('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    a_label = 'Baseline (rollout = 1)'

    seed_list = torch.tensor([105, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 
                 42, 65, 82])

    topk_models_converged_loss = []
    topk_models = []
    for seed in seed_list:
        b = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
        topk_models.append(b) 
        topk_models_converged_loss.append(b['loss_hist_train'][-1])

    # Re-order based on converged loss
    _, sort_idx = torch.sort(torch.tensor(topk_models_converged_loss))
    seed_list = seed_list[sort_idx] 
    
    # Re-read: 
    topk_models_converged_loss = []
    topk_models = []
    for seed in seed_list:
        b = torch.load('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))
        topk_models.append(b) 
        topk_models_converged_loss.append(b['loss_hist_train'][-1])

    # Combined loss plot 
    baseline_loss = np.mean(a['loss_hist_train'][-10:])

    fig, ax = plt.subplots()
    ax.axhline(y=baseline_loss, color='black', linestyle='--', lw=2)
    for i in range(len(seed_list)):
        ax.plot(topk_models[i]['loss_hist_train'][1:], lw=2, zorder=-1)
    #ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Epochs')
    plt.show(block=False)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Baseline error budget: what percent of baseline error is in masked region? 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0: 
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    # Read data: 
    if 1 == 1:
        modelname_list = []
        #modelname_list = ['topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0']

        # If seed_list has not already been initialized:
        if seed_list == None:
            seed_list = torch.tensor([105, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 
                 42, 65, 82])
        mse_mask_list = []
        mse_full_list = []
        for seed in seed_list:
            mse_mask = np.load('outputs/postproc/mse_mask_budget_data/Re_32564/mse_mask_pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy' %(seed))[:, 2:, :]
            mse_mask_list.append(mse_mask)

            mse_full = np.load('outputs/postproc/mse_mask_budget_data/Re_32564/mse_full_pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy' %(seed))[:, 2:, :]
            mse_full_list.append(mse_full)
        
        rollout_id = 0

        # Percentage: 
        fig, ax = plt.subplots(3,5, figsize=(16,8), sharex=True, sharey=False)
        for r in range(3):
            for c in range(5):
                seed_id = r*5 + c
                for comp in range(2):
                    mse_full = mse_full_list[seed_id][rollout_id,:,comp]
                    mse_mask = mse_mask_list[seed_id][rollout_id,:,comp]
                    mse_not_mask = mse_full - mse_mask
                    percent_mask = (mse_mask/mse_full)*100
                    percent_not_mask = (mse_not_mask/mse_full)*100
                    ax[r,c].plot(percent_mask)
                    #ax[r,c].plot(percent_not_mask, color='black')
                ax[r,c].set_title('Seed = %d' %(seed_list[seed_id]))
                ax[r,c].set_ylim([0,100])
        
        #ax.set_ylabel('MSE Budget [%]')
        #ax.set_xlabel('Time')
        plt.show(block=False)


        # Just plot one 
        seed_id = 5
        fig, ax = plt.subplots()
        for comp in range(2):
            mse_full = mse_full_list[seed_id][rollout_id,:,comp]
            mse_mask = mse_mask_list[seed_id][rollout_id,:,comp]
            mse_not_mask = mse_full - mse_mask
            percent_mask = (mse_mask/mse_full)*100
            percent_not_mask = (mse_not_mask/mse_full)*100
            ax.plot(percent_mask)
        ax.set_title('Seed = %d' %(seed_list[seed_id]))
        ax.set_ylim([0,100])
        ax.set_ylabel('MSE Budget [%]')
        ax.set_xlabel('Time Step')
        plt.show(block=False)


    # Write data: 
    if 1 == 0: 
        # Load baseline model 
        modelpath_baseline = './saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
        p = torch.load(modelpath_baseline)
        input_dict = p['input_dict']
        model_baseline = gnn.GNN_TopK_NoReduction(
                    in_channels_node = input_dict['in_channels_node'],
                    in_channels_edge = input_dict['in_channels_edge'],
                    hidden_channels = input_dict['hidden_channels'],
                    out_channels = input_dict['out_channels'], 
                    n_mlp_encode = input_dict['n_mlp_encode'], 
                    n_mlp_mp = input_dict['n_mlp_mp'],
                    n_mp_down_topk = input_dict['n_mp_down_topk'],
                    n_mp_up_topk = input_dict['n_mp_up_topk'],
                    pool_ratios = input_dict['pool_ratios'], 
                    n_mp_down_enc = input_dict['n_mp_down_enc'], 
                    n_mp_up_enc = input_dict['n_mp_up_enc'], 
                    n_mp_down_dec = input_dict['n_mp_down_dec'], 
                    n_mp_up_dec = input_dict['n_mp_up_dec'], 
                    lengthscales_enc = input_dict['lengthscales_enc'],
                    lengthscales_dec = input_dict['lengthscales_dec'], 
                    bounding_box = input_dict['bounding_box'], 
                    interpolation_mode = input_dict['interp'], 
                    act = input_dict['act'], 
                    param_sharing = input_dict['param_sharing'],
                    filter_lengthscale = input_dict['filter_lengthscale'], 
                    name = input_dict['name'])
        model_baseline.load_state_dict(p['state_dict'])
        model_baseline.to(device)
        model_baseline.eval()

        seed_list = [105, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 
                 42, 65, 82]

        for seed in seed_list:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('SEED %d' %(seed))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            modelpath_topk = 'saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed) 
            p = torch.load(modelpath_topk)
            input_dict = p['input_dict']
            model_topk = gnn.GNN_TopK_NoReduction(
                    in_channels_node = input_dict['in_channels_node'],
                    in_channels_edge = input_dict['in_channels_edge'],
                    hidden_channels = input_dict['hidden_channels'],
                    out_channels = input_dict['out_channels'], 
                    n_mlp_encode = input_dict['n_mlp_encode'], 
                    n_mlp_mp = input_dict['n_mlp_mp'],
                    n_mp_down_topk = input_dict['n_mp_down_topk'],
                    n_mp_up_topk = input_dict['n_mp_up_topk'],
                    pool_ratios = input_dict['pool_ratios'], 
                    n_mp_down_enc = input_dict['n_mp_down_enc'], 
                    n_mp_up_enc = input_dict['n_mp_up_enc'], 
                    n_mp_down_dec = input_dict['n_mp_down_dec'], 
                    n_mp_up_dec = input_dict['n_mp_up_dec'], 
                    lengthscales_enc = input_dict['lengthscales_enc'],
                    lengthscales_dec = input_dict['lengthscales_dec'], 
                    bounding_box = input_dict['bounding_box'], 
                    interpolation_mode = input_dict['interp'], 
                    act = input_dict['act'], 
                    param_sharing = input_dict['param_sharing'],
                    filter_lengthscale = input_dict['filter_lengthscale'], 
                    name = input_dict['name'])
            model_topk.load_state_dict(p['state_dict'])
            model_topk.to(device)
            model_topk.eval()

            # ~~~~ Re-load data: 
            rollout_eval = 1 # where to evaluate the RMSE  
            #vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_32564.vtk'
            vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_26214.vtk'
            path_to_ei = 'datasets/BACKWARD_FACING_STEP/edge_index'
            path_to_ea = 'datasets/BACKWARD_FACING_STEP/edge_attr'
            path_to_pos = 'datasets/BACKWARD_FACING_STEP/pos'
            device_for_loading = device

            dataset_eval, _ = bfs.get_pygeom_dataset_cell_data_radius(
                            vtk_file_test, 
                            path_to_ei, 
                            path_to_ea,
                            path_to_pos, 
                            device_for_loading, 
                            time_lag = rollout_eval,
                            scaling = [data_mean, data_std],
                            features_to_keep = [1,2], 
                            fraction_valid = 0, 
                            multiple_cases = False)

            # Remove first snapshot
            test_dataset.pop(0)

            # Loop through test snapshots 
            N = len(dataset_eval)

            # populate RMSE versus time plot 
            mse_full = []
            mse_mask = [] # error in the masked region 
            for i in range(rollout_eval):
                mse_full.append(np.zeros((N,2))) # 2 components for ux, uy
                mse_mask.append(np.zeros((N,2))) # 2 components for ux, uy

            for i in range(N): 
                print('Snapshot %d/%d' %(i+1, N))
                data = dataset_eval[i]
                x_new = data.x
                for t in range(rollout_eval):
                    print('\tRollout %d/%d' %(t+1, rollout_eval))

                    # Get prediction from baseline 
                    x_old = torch.clone(x_new)
                    x_src = model_baseline(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
                    x_new = x_old + x_src

                    # Get mask from topk
                    mask = model_topk.get_mask(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
                    # fig, ax = plt.subplots()
                    # ax.scatter(data.pos[:,0], data.pos[:,1], c=mask)
                    # plt.show(block=False)

                    target = data.y[t]

                    # Compute MSE budget
                    n_nodes = target.shape[0]
                    mse_total_0 = (1.0/n_nodes) * torch.sum( (x_new[:,0] - target[:,0])**2 ) 
                    mse_mask_0 = (1.0/n_nodes) * torch.sum( mask * ((x_new[:,0] - target[:,0])**2) )

                    mse_total_1 = (1.0/n_nodes) * torch.sum( (x_new[:,1] - target[:,1])**2 ) 
                    mse_mask_1 = (1.0/n_nodes) * torch.sum( mask * ((x_new[:,1] - target[:,1])**2) )

                    # Store 
                    mse_full[t][i][0] = mse_total_0 
                    mse_full[t][i][1] = mse_total_1 

                    mse_mask[t][i][0] = mse_mask_0 
                    mse_mask[t][i][1] = mse_mask_1 

            # Save mse data
            print('SAVING...')
            print('SAVING...')
            print('SAVING...')
            savepath = './outputs/postproc/mse_mask_budget_data'
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            np.save(savepath + '/mse_full_%s.npy' %(model_topk.get_save_header()), mse_full)
            np.save(savepath + '/mse_mask_%s.npy' %(model_topk.get_save_header()), mse_mask)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Postprocess testing losses: RMSE  
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0: 
    print('postprocess testing losses.')

    # set device 
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # # Load models: effect of rollout length 
    # modelpath_list = []
    # modelpath_list.append('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # modelpath_list.append('saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_1_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # modelpath_list.append('saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_2_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # modelpath_list.append('saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_3_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # modelpath_list.append('saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_4_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    # modelpath_list.append('saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_5_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')

    # Load models: effect of seed for R = 1
    modelpath_list = []
    modelpath_list.append('saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    
    if seed_list == None:
        seed_list = [105, 122, 132, 142, 152, 162, 172, 182, 192, 202, 212, 222, 
                 42, 65, 82]
    for seed in seed_list:
        modelpath_list.append('saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed))

    # Load rmse data -- effect of seed: 
    if 1 == 0:
        rmse_path = './outputs/postproc/rmse_data/Re_26214/'
        #rmse_path = './outputs/postproc/rmse_data/Re_32564/'

        rmse_baseline = np.load(rmse_path + 'topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
       
        rmse_topk_mean = []
        for seed in seed_list:
            rmse_topk_seed = np.load(rmse_path + 'pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy' %(seed))
            rmse_topk_mean.append(np.mean(rmse_topk_seed, axis=1))

        # compute average rmse:
        rmse_baseline_mean = np.mean(rmse_baseline, axis=1)
        rmse_baseline_max = np.max(rmse_baseline, axis=1)
        rmse_baseline_min = np.min(rmse_baseline, axis=1)
        
        rollout_steps = np.arange(1,rmse_baseline.shape[0] + 1)

        # Fig, ax 
        lw = 2
        ms = 1
        mew = 1
        fig, ax = plt.subplots(1,2, figsize=(14,5))

        # ux: 
        ax[0].plot(rollout_steps, rmse_baseline_mean[:,0], label='Baseline', lw=lw, color='black')
        for seed_id in range(len(seed_list)):
            ax[0].plot(rollout_steps, rmse_topk_mean[seed_id][:,0], lw=lw, color='red', zorder=-1)
        ax[0].set_xlabel('Rollout Steps') 
        ax[0].set_ylabel('RMSE')
        ax[0].set_title('Ux')

        # uy:
        ax[1].plot(rollout_steps, rmse_baseline_mean[:,1], label='Baseline', lw=lw, color='black')
        for seed_id in range(len(seed_list)):
            ax[1].plot(rollout_steps, rmse_topk_mean[seed_id][:,1], lw=lw, color='red', zorder=-1)
        ax[1].set_xlabel('Rollout Steps') 
        ax[1].set_ylabel('RMSE')
        ax[1].set_title('Uy')

        #ax[0].set_ylim([1e-3, 1e-1])
        #ax[1].set_ylim([1e-3, 1e-1])
        #ax[0].set_yscale('log')
        #ax[1].set_yscale('log')
        #ax[0].legend(framealpha=1, fancybox=False, edgecolor='black', prop={'size': 14})
        plt.show(block=False)

    # Load rmse data -- effect of rollout length (no seed): 
    if 1 == 0:
        rmse_path = './outputs/postproc/rmse_data/Re_26214/'
        #rmse_path = './outputs/postproc/rmse_data/Re_32564/'

        rmse_baseline = np.load(rmse_path + 'topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
        rmse_topk1 = np.load(rmse_path + 'pretrained_topk_unet_rollout_1_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
        rmse_topk2 = np.load(rmse_path + 'pretrained_topk_unet_rollout_2_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
        rmse_topk3 = np.load(rmse_path + 'pretrained_topk_unet_rollout_3_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
        rmse_topk4 = np.load(rmse_path + 'pretrained_topk_unet_rollout_4_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')
        rmse_topk5 = np.load(rmse_path + 'pretrained_topk_unet_rollout_5_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.npy')


        # compute average rmse:
        rmse_baseline_mean = np.mean(rmse_baseline, axis=1)
        rmse_baseline_max = np.max(rmse_baseline, axis=1)
        rmse_baseline_min = np.min(rmse_baseline, axis=1)

        rmse_topk1_mean = np.mean(rmse_topk1, axis=1)
        rmse_topk2_mean = np.mean(rmse_topk2, axis=1)
        rmse_topk3_mean = np.mean(rmse_topk3, axis=1)
        rmse_topk4_mean = np.mean(rmse_topk4, axis=1)
        rmse_topk5_mean = np.mean(rmse_topk5, axis=1)

        rollout_steps = np.arange(1,rmse_baseline.shape[0] + 1)


        # Fig, ax 
        lw = 2
        ms = 1
        mew = 1
        fig, ax = plt.subplots(1,2, figsize=(14,5))

        # ux: 
        ax[0].plot(rollout_steps, rmse_baseline_mean[:,0], label='Baseline', lw=lw)
        ax[0].plot(rollout_steps, rmse_topk1_mean[:,0], label='TopK, R=1', lw=lw)
        ax[0].plot(rollout_steps, rmse_topk2_mean[:,0], label='TopK, R=2', lw=lw)
        ax[0].plot(rollout_steps, rmse_topk3_mean[:,0], label='TopK, R=3', lw=lw)
        ax[0].plot(rollout_steps, rmse_topk4_mean[:,0], label='TopK, R=4', lw=lw)
        ax[0].plot(rollout_steps, rmse_topk5_mean[:,0], label='TopK, R=5', lw=lw)
        ax[0].set_xlabel('Rollout Steps') 
        ax[0].set_ylabel('RMSE')
        ax[0].set_title('Ux')

        # uy:
        ax[1].plot(rollout_steps, rmse_baseline_mean[:,1], label='Baseline', lw=lw)
        ax[1].plot(rollout_steps, rmse_topk1_mean[:,1], label='TopK, R=1', lw=lw)
        ax[1].plot(rollout_steps, rmse_topk2_mean[:,1], label='TopK, R=2', lw=lw)
        ax[1].plot(rollout_steps, rmse_topk3_mean[:,1], label='TopK, R=3', lw=lw)
        ax[1].plot(rollout_steps, rmse_topk4_mean[:,1], label='TopK, R=4', lw=lw)
        ax[1].plot(rollout_steps, rmse_topk5_mean[:,1], label='TopK, R=5', lw=lw)
        ax[1].set_xlabel('Rollout Steps') 
        ax[1].set_ylabel('RMSE')
        ax[1].set_title('Uy')

        ax[0].set_ylim([1e-3, 1e-1])
        ax[1].set_ylim([1e-3, 1e-1])
        ax[0].set_yscale('log')
        ax[1].set_yscale('log')
        ax[0].legend(framealpha=1, fancybox=False, edgecolor='black', prop={'size': 14})
        plt.show(block=False)

    # Write data: 
    if 1 == 1: 
        for modelpath in modelpath_list:
            p = torch.load(modelpath)
            input_dict = p['input_dict']

            # with top-k, no reduction
            model = gnn.GNN_TopK_NoReduction(
                    in_channels_node = input_dict['in_channels_node'],
                    in_channels_edge = input_dict['in_channels_edge'],
                    hidden_channels = input_dict['hidden_channels'],
                    out_channels = input_dict['out_channels'], 
                    n_mlp_encode = input_dict['n_mlp_encode'], 
                    n_mlp_mp = input_dict['n_mlp_mp'],
                    n_mp_down_topk = input_dict['n_mp_down_topk'],
                    n_mp_up_topk = input_dict['n_mp_up_topk'],
                    pool_ratios = input_dict['pool_ratios'], 
                    n_mp_down_enc = input_dict['n_mp_down_enc'], 
                    n_mp_up_enc = input_dict['n_mp_up_enc'], 
                    n_mp_down_dec = input_dict['n_mp_down_dec'], 
                    n_mp_up_dec = input_dict['n_mp_up_dec'], 
                    lengthscales_enc = input_dict['lengthscales_enc'],
                    lengthscales_dec = input_dict['lengthscales_dec'], 
                    bounding_box = input_dict['bounding_box'], 
                    interpolation_mode = input_dict['interp'], 
                    act = input_dict['act'], 
                    param_sharing = input_dict['param_sharing'],
                    filter_lengthscale = input_dict['filter_lengthscale'], 
                    name = input_dict['name'])

            model.load_state_dict(p['state_dict'])
            model.to(device)
            model.eval()

            asdf

            # ~~~~ Re-load data: 
            rollout_eval = 10 # where to evaluate the RMSE  
            #vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_32564.vtk'
            vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_26214.vtk'
            path_to_ei = 'datasets/BACKWARD_FACING_STEP/edge_index'
            path_to_ea = 'datasets/BACKWARD_FACING_STEP/edge_attr'
            path_to_pos = 'datasets/BACKWARD_FACING_STEP/pos'
            device_for_loading = device

            dataset_eval_rmse, _ = bfs.get_pygeom_dataset_cell_data(
                            vtk_file_test, 
                            path_to_ei, 
                            path_to_ea,
                            path_to_pos, 
                            device_for_loading, 
                            time_lag = rollout_eval,
                            scaling = [data_mean, data_std],
                            features_to_keep = [1,2], 
                            fraction_valid = 0, 
                            multiple_cases = False)

            # Remove first snapshot
            test_dataset.pop(0)

            # Loop through test snapshots 
            N = len(dataset_eval_rmse)

            # populate RMSE versus time plot 
            rmse_data = []
            for i in range(rollout_eval):
                rmse_data.append(np.zeros((N,2))) # 2 components for ux, uy

            for i in range(N): 
                print('Snapshot %d/%d' %(i+1, N))
                data = dataset_eval_rmse[i]
                x_new = data.x
                for t in range(rollout_eval):
                    print('\tRollout %d/%d' %(t+1, rollout_eval))
                    x_old = torch.clone(x_new)
                    x_src = model(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
                    x_new = x_old + x_src

                    # Accumulate loss 
                    target = data.y[t]

                    # compute rmse 
                    rmse_data[t][i][0] = torch.sqrt(F.mse_loss(x_new[:,0], target[:,0]))
                    rmse_data[t][i][1] = torch.sqrt(F.mse_loss(x_new[:,1], target[:,1]))


            # Save rmse_data
            print('SAVING...')
            print('SAVING...')
            print('SAVING...')
            savepath = './outputs/postproc/rmse_data'
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            np.save(savepath + '/%s.npy' %(model.get_save_header()), rmse_data)
            print('Saved at: %s' %(savepath + '/%s.npy' %(model.get_save_header())))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Write model predictions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    #modelpath = 'saved_models/model_multi_scale.tar'
    #modelpath = 'saved_models/model_single_scale.tar'

    #modelpath = 'saved_models/topk_unet_rollout_1_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
    #modelpath = 'saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_1_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
    #modelpath = 'saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_2_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
    #modelpath = 'saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_3_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
    #modelpath = 'saved_models/pretrained_seed_1/pretrained_topk_unet_rollout_5_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'


    seed = 182
    modelpath = 'saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed)

    p = torch.load(modelpath)

    input_dict = p['input_dict']
    print('input_dict: ', input_dict)

    # with top-k, no reduction
    model = gnn.GNN_TopK_NoReduction(
            in_channels_node = input_dict['in_channels_node'],
            in_channels_edge = input_dict['in_channels_edge'],
            hidden_channels = input_dict['hidden_channels'],
            out_channels = input_dict['out_channels'], 
            n_mlp_encode = input_dict['n_mlp_encode'], 
            n_mlp_mp = input_dict['n_mlp_mp'],
            n_mp_down_topk = input_dict['n_mp_down_topk'],
            n_mp_up_topk = input_dict['n_mp_up_topk'],
            pool_ratios = input_dict['pool_ratios'], 
            n_mp_down_enc = input_dict['n_mp_down_enc'], 
            n_mp_up_enc = input_dict['n_mp_up_enc'], 
            n_mp_down_dec = input_dict['n_mp_down_dec'], 
            n_mp_up_dec = input_dict['n_mp_up_dec'], 
            lengthscales_enc = input_dict['lengthscales_enc'],
            lengthscales_dec = input_dict['lengthscales_dec'], 
            bounding_box = input_dict['bounding_box'], 
            interpolation_mode = input_dict['interp'], 
            act = input_dict['act'], 
            param_sharing = input_dict['param_sharing'],
            filter_lengthscale = input_dict['filter_lengthscale'], 
            name = input_dict['name'])

    model.load_state_dict(p['state_dict'])
    model.eval()

    # Plot the losses:
    fig, ax = plt.subplots()
    colors = ['black','blue','red']
    ax.plot(p['loss_hist_train'], label='train', lw=2, color='black')
    ax.plot(p['loss_hist_test'], label='valid', lw=2, color='black', ls='--')

    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.grid(True)
    #ax.set_ylim([1e-6, 1])
    ax.set_ylim([1e-4, 0.1]) 

    ax.legend(fancybox=False, framealpha=1, edgecolor='black')
    ax.set_title('Training History')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    plt.show(block=False)

    # Set to eval mode
    model.eval()
    header = model.get_save_header()

    # Get Re:
    str_temp = vtk_file_test.split('.')[0]
    str_re = str_temp[-8:]

    # Update save directory with trajectory index. This is where openfoam cases will be saved. 
    save_dir = '/Users/sbarwey/Files/openfoam_cases/backward_facing_step/Backward_Facing_Step_Cropped_Predictions_Forecasting/%s/' %(str_re)

    if not  os.path.exists(save_dir + '/' + header):
        os.makedirs(save_dir + '/' + header)

    # Get input 
    n_nodes =  test_dataset[0].x.shape[0]
    n_features = test_dataset[0].x.shape[1]
    field_names = ['ux', 'uy']
    #u_vec_target = np.zeros((n_nodes,3))
    #u_vec_pred = np.zeros((n_nodes,3))

    ic_index = 240 # 120
    x_new = test_dataset[ic_index].x
    for i in range(ic_index,len(test_dataset)):
        print('[%d/%d]' %(i+1, len(test_dataset)))
        data = test_dataset[i]

        # Get time 
        time_value = data.t.item()

        # Get single step prediction
        print('\tSingle step...')
        x_src = model.forward(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
        x_new_singlestep = data.x + x_src

        # Get mask (single-step): 
        print('\tMask single step...')
        mask_singlestep = model.get_mask(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)

        # Get rollout prediction
        print('\tRollout step...')
        x_old = torch.clone(x_new)
        x_src = model.forward(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
        x_new = x_old + x_src
        target = data.y[0]

        print('\tMask rollout step...')
        mask_rollout = model.get_mask(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)

        # unscale target and prediction 
        mean_i = data.data_scale[0].reshape((1,n_features))
        std_i = data.data_scale[1].reshape((1,n_features))
        x_old_unscaled = x_old * std_i + mean_i
        x_new_unscaled = x_new * std_i + mean_i
        x_new_singlestep_unscaled = x_new_singlestep * std_i + mean_i
        target_unscaled = target * std_i + mean_i

        print('\tRollout error...')
        error_rollout = x_new_unscaled  - target_unscaled
        error_singlestep = x_new_singlestep_unscaled - target_unscaled
        
        # Create time folder 
        time_folder = save_dir + '/' + header + '/' + '%g' %(time_value)
        if not os.path.exists(time_folder):
            os.makedirs(time_folder)

        # Write data to time folder 
        for f in range(n_features):

            # Prediction singlestep   
            field_name = '%s_pred_singlestep' %(field_names[f])
            scalar2openfoam(x_new_singlestep_unscaled[:,f].numpy(), 
                            time_folder+'/%s' %(field_name), field_name, time_value)


            # Prediction rollout
            field_name = '%s_pred_rollout' %(field_names[f])
            scalar2openfoam(x_new_unscaled[:,f].numpy(), 
                            time_folder+'/%s' %(field_name), field_name, time_value)

            # Target 
            field_name = '%s_target' %(field_names[f])
            scalar2openfoam(target_unscaled[:,f].numpy(), 
                            time_folder+'/%s' %(field_name), field_name, time_value)

            # Error rollout  
            field_name = '%s_error_rollout' %(field_names[f])
            scalar2openfoam(error_rollout[:,f].numpy(), 
                            time_folder+'/%s' %(field_name), field_name, time_value)

            # Error singlestep  
            field_name = '%s_error_singlestep' %(field_names[f])
            scalar2openfoam(error_singlestep[:,f].numpy(), 
                            time_folder+'/%s' %(field_name), field_name, time_value)


        # # Velocity vector
        # u_vec_target[:,:2] = target[:,:]
        # u_vec_pred[:,:2] = pred[:,:]
        # u_vec_error = pred - target

        # field_name = 'U_target'
        # array2openfoam(u_vec_target,  time_folder+'/%s' %(field_name), field_name, time_value)

        # field_name = 'U_pred'
        # array2openfoam(u_vec_pred,  time_folder+'/%s' %(field_name), field_name, time_value)

        # mask singlestep
        field_name = 'mask_singlestep'
        scalar2openfoam(mask_singlestep.numpy().squeeze(), time_folder+'/%s' %(field_name), field_name, time_value)

        # mask rollout
        field_name = 'mask_rollout'
        scalar2openfoam(mask_rollout.numpy().squeeze(), time_folder+'/%s' %(field_name), field_name, time_value)


# ~~~~ Plot time evolution at the sensors
if 1 == 0:
    #case_dir = '/Users/sbarwey/Files/openfoam_cases/backward_facing_step/Backward_Facing_Step_Cropped_Predictions_Forecasting/Re_32564/model_multi_scale'
    case_dir = '/Users/sbarwey/Files/openfoam_cases/backward_facing_step/Backward_Facing_Step_Cropped_Predictions_Forecasting/Re_32564/model_single_scale'
    #case_dir = '/Users/sbarwey/Files/openfoam_cases/backward_facing_step/Backward_Facing_Step_Cropped_Predictions_Forecasting/Re_32564/topk_down_topk_1_1_up_topk_1_1_factor_16_hc_128_down_enc_4_up_enc_down_dec_4_4_4_up_dec_4_4_param_sharing_0'

    sensor_dir = case_dir + '/postProcessing/probes1/'

    field = 'uy'
    step_max = 40
    data_target         = np.loadtxt(sensor_dir + '%s_target' %(field))[:step_max]
    data_singlestep     = np.loadtxt(sensor_dir + '%s_pred_singlestep'%(field))[:step_max]
    data_rollout        = np.loadtxt(sensor_dir + '%s_pred_rollout'%(field))[:step_max]

    time_vec = data_target[:,0]
    time_vec = time_vec - time_vec[0]

    time_vec = np.arange(1,len(time_vec)+1)

    wake_1 = 1
    wake_2 = 2
    wake_3 = 3
    fs_1 = 4
    fs_2 = 5
    fs_3 = 6

    lw = 1.5
    ms = 8
    me = 1
    marker_singlestep = 'o'
    marker_rollout = 's'

    fig, ax = plt.subplots(2,3, figsize=(15,8))
    ax[0,0].plot(time_vec, data_target[:,wake_1], color='black', lw=lw, label='Target')
    ax[0,0].plot(time_vec, data_singlestep[:,wake_1], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[0,0].plot(time_vec, data_rollout[:,wake_1], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[0,0].set_xlabel('Steps')
    ax[0,0].set_ylabel('%s' %(field))
    ax[0,0].set_ylim([data_target[:,wake_1].min() - 0.50, data_target[:,wake_1].max() + 0.50])

    ax[0,1].plot(time_vec, data_target[:,wake_2], color='black', lw=lw, label='Target')
    ax[0,1].plot(time_vec, data_singlestep[:,wake_2], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[0,1].plot(time_vec, data_rollout[:,wake_2], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[0,1].set_xlabel('Steps')
    ax[0,1].set_ylabel('%s' %(field))
    ax[0,1].set_ylim([data_target[:,wake_2].min() - 0.50, data_target[:,wake_2].max() + 0.50])


    ax[0,2].plot(time_vec, data_target[:,wake_3], color='black', lw=lw, label='Target')
    ax[0,2].plot(time_vec, data_singlestep[:,wake_3], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[0,2].plot(time_vec, data_rollout[:,wake_3], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[0,2].set_xlabel('Steps')
    ax[0,2].set_ylabel('%s' %(field))
    ax[0,2].set_ylim([data_target[:,wake_3].min() - 0.50, data_target[:,wake_3].max() + 0.50])

    ax[1,0].plot(time_vec, data_target[:,fs_1], color='black', lw=lw, label='Target')
    ax[1,0].plot(time_vec, data_singlestep[:,fs_1], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[1,0].plot(time_vec, data_rollout[:,fs_1], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[1,0].set_xlabel('Steps')
    ax[1,0].set_ylabel('%s' %(field))
    ax[1,0].set_ylim([data_target[:,fs_1].min() - 0.50, data_target[:,fs_1].max() + 0.50])

    ax[1,1].plot(time_vec, data_target[:,fs_2], color='black', lw=lw, label='Target')
    ax[1,1].plot(time_vec, data_singlestep[:,fs_2], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[1,1].plot(time_vec, data_rollout[:,fs_2], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[1,1].set_xlabel('Steps')
    ax[1,1].set_ylabel('%s' %(field))
    ax[1,1].set_ylim([data_target[:,fs_2].min() - 0.50, data_target[:,fs_2].max() + 0.50])


    ax[1,2].plot(time_vec, data_target[:,fs_3], color='black', lw=lw, label='Target')
    ax[1,2].plot(time_vec, data_singlestep[:,fs_3], color='blue', ls='--', lw=lw,
                marker=marker_singlestep, ms=ms, fillstyle='none', markevery=me, label='Single Step')
    ax[1,2].plot(time_vec, data_rollout[:,fs_3], color='red', ls='--', lw=lw,
                marker=marker_rollout, ms=ms, fillstyle='none', markevery=me, label='Rollout')
    ax[1,2].set_xlabel('Steps')
    ax[1,2].set_ylabel('%s' %(field))
    ax[1,2].set_ylim([data_target[:,fs_3].min() - 0.50, data_target[:,fs_3].max() + 0.50])
    ax[1,2].legend(fancybox=False, framealpha=1, edgecolor='black')

    plt.show()





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize MMP blocks from a pre-trained model 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:

    # Step 1: Load previous top-k model
    modelpath = 'saved_models/pretrained_topk_unet_rollout_1_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar'
    p = torch.load(modelpath)

    input_dict = p['input_dict']
    print('input_dict: ', input_dict)
    
    # Step 2: load new top-k model
    bbox = input_dict['bounding_box']
    model_2 = gnn.GNN_TopK_NoReduction(
              in_channels_node = input_dict['in_channels_node'],
              in_channels_edge = input_dict['in_channels_edge'],
              hidden_channels = input_dict['hidden_channels'],
              out_channels = input_dict['out_channels'], 
              n_mlp_encode = input_dict['n_mlp_encode'], 
              n_mlp_mp = input_dict['n_mlp_mp'],
              n_mp_down_topk = input_dict['n_mp_down_topk'],
              n_mp_up_topk = input_dict['n_mp_up_topk'],
              pool_ratios = input_dict['pool_ratios'], 
              n_mp_down_enc = input_dict['n_mp_down_enc'], 
              n_mp_up_enc = input_dict['n_mp_up_enc'], 
              n_mp_down_dec = input_dict['n_mp_down_dec'], 
              n_mp_up_dec = input_dict['n_mp_up_dec'], 
              lengthscales_enc = input_dict['lengthscales_enc'],
              lengthscales_dec = input_dict['lengthscales_dec'], 
              bounding_box = input_dict['bounding_box'], 
              interpolation_mode = input_dict['interp'], 
              act = input_dict['act'], 
              param_sharing = input_dict['param_sharing'],
              filter_lengthscale = input_dict['filter_lengthscale'], 
              name = 'gnn_topk_2')

    # Load state dict from the previous top-k model
    model_2.load_state_dict(p['state_dict'])


    # Freeze all params except the top-k and the MMP param

    # print number of params before over-writing: 
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('number of parameters before overwriting: ', count_parameters(model_2))
    print('number of parameters before overwriting: ', count_parameters(model_2))
    print('number of parameters before overwriting: ', count_parameters(model_2))
    
    # Write params 
    model_2.set_mmp_layer(model_2.down_mps[0][0], model_2.down_mps[0][0])
    model_2.set_mmp_layer(model_2.up_mps[0][0], model_2.up_mps[0][0])
        

    # print number of params after over-writing:
    print('number of parameters after overwriting: ', count_parameters(model_2))
    print('number of parameters after overwriting: ', count_parameters(model_2))
    print('number of parameters after overwriting: ', count_parameters(model_2))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Gradient tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    
    # Read-in the graph
    vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_32564.vtk'
    path_to_ei = 'datasets/BACKWARD_FACING_STEP/edge_index'
    path_to_ea = 'datasets/BACKWARD_FACING_STEP/edge_attr'
    path_to_pos = 'datasets/BACKWARD_FACING_STEP/pos'
    path_to_edge_sign = 'datasets/BACKWARD_FACING_STEP/edge_sign'
    path_to_interp_weights = 'datasets/BACKWARD_FACING_STEP/interp_weights'
    path_to_pos_face = 'datasets/BACKWARD_FACING_STEP/pos_face'
    path_to_vol = 'datasets/BACKWARD_FACING_STEP/cell_volume'
    path_to_sf = 'datasets/BACKWARD_FACING_STEP/sf_normal'
    path_to_node_type = 'datasets/BACKWARD_FACING_STEP/node_type'
    device_for_loading = 'cpu'
        

    import pyvista as pv
    mesh = pv.read(vtk_file_test)
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

    # Edge attributes and index, and node positions 
      

    time_lag = 0
    n_snaps = data_full.shape[0] - time_lag
    data_x = []
    data_y = []
    for i in range(n_snaps):
        data_x.append([data_full[i]])
        if time_lag == 0:
            y_temp = [data_full[i]]
        else:
            y_temp = []
            for t in range(1, time_lag+1):
                y_temp.append(data_full[i+t])
        data_y.append(y_temp)

    data_x = np.array(data_x) # shape: [n_snaps, 1, n_nodes, n_features]
    data_y = np.array(data_y) # shape: [n_snaps, time_lag, n_nodes, n_features]

    # data = Data( x = data_x, pos = pos, edge_index = edge_index )

    # Load the other stuff 
    edge_index = torch.tensor(np.loadtxt(path_to_ei, dtype=np.long).T)
    pos = torch.tensor(np.loadtxt(path_to_pos, dtype=np.float32))
    edge_sign = torch.tensor(np.loadtxt(path_to_edge_sign, dtype=np.float32))
    interp_weights = torch.tensor(np.loadtxt(path_to_interp_weights, dtype=np.float32))
    pos_face = torch.tensor(np.loadtxt(path_to_pos_face, dtype=np.float32))
    vol = torch.tensor(np.loadtxt(path_to_vol, dtype=np.float32))
    sf = torch.tensor(np.loadtxt(path_to_sf, dtype=np.float32))
    node_type = torch.tensor(np.loadtxt(path_to_node_type, dtype=np.float32))


    # ~~~~ Step 1: node-to-edge interpolation 
    # Get the data to interpolate
    #phi = torch.tensor(data_x[10, 0, :, 0]) # 10th snapshot, pressure field
    phi = torch.tensor(data_x[10, 0, :, 1]) # 10th snapshot, ux field
    #phi = torch.tensor(data_x[10, 0, :, 2]) # 10th snapshot, uy field
    
    # Get the owner nodes 
    n_edges = edge_index.shape[1]

    # Get node indices 
    node_nei = edge_index[0,:]
    node_own = edge_index[1,:]

    # Get node values for these indices  
    pos_nei = pos[node_nei]
    pos_own = pos[node_own]

    phi_nei = phi[node_nei].view(-1,1)
    phi_own = phi[node_own].view(-1,1)


    # Get face position 
    pos_e = pos_face[:,:2]

    # # Re-compute weights 
    # w = (pos_e - pos_nei)/(pos_own - pos_nei)

    # Interpolate the position 
    # interp = w_limiter_pos * (surf_own[j] - surf_nei[j]) + surf_nei[j]
    test = interp_weights.view(-1,1) * (pos_own - pos_nei) + pos_nei 
    phi_f = interp_weights.view(-1,1) * (phi_own - phi_nei) + phi_nei

    # ~~~~ Step 2: Do the surface integration  
    # -- GRAD = ( 1 / vol ) * \sum phi_f * S_f * n_f [INCREASES DIMENSIONALITY] -- there should be no gradient in z 
    edge_attr = phi_f * (sf * edge_sign.view(-1,1))
   
    # sum the edge attributes 
    surface_int = gnn.EdgeAggregation(aggr='add')
    gradient = (1.0/vol.view(-1,1))*surface_int(pos, edge_index, edge_attr) 
    #gradient = surface_int(pos, edge_index, edge_attr) 

    # Zero the boundary gradients 
    gradient = gradient * node_type.view(-1,1)

    # plot 
    idx_plot = node_type == 1

    
    vmax = 1e4 # ux 
    #vmax = 5e3 # uy 
    vmin = -vmax
    ms = 5
    fig,ax = plt.subplots(1,3,figsize=(18,5), sharex=True, sharey=True)
    ax[0].scatter(pos[idx_plot,0], pos[idx_plot,1], c=phi[idx_plot], vmin=-30, vmax=30, s=ms)
    ax[1].scatter(pos[idx_plot,0], pos[idx_plot,1], c=gradient[idx_plot,0], vmin=vmin, vmax=vmax, s=ms)
    ax[2].scatter(pos[idx_plot,0], pos[idx_plot,1], c=gradient[idx_plot,1], vmin=vmin, vmax=vmax, s=ms)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')
    plt.show(block=False)
    


    # ~~~~ Other option: edge-based total variation
    edge_variation = torch.abs(phi_own - phi_nei)
    total_variation = torch.sum(edge_variation) 






# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Developing the mask loss function
# GOAL :
# We want to minimize the error outside of the mask! 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 1: 
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'


    seed = 105 
    modelpath_topk = 'saved_models/pretrained_topk_unet_rollout_1_seed_%d_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar' %(seed) 
    p = torch.load(modelpath_topk)
    input_dict = p['input_dict']
    model_topk = gnn.GNN_TopK_NoReduction(
            in_channels_node = input_dict['in_channels_node'],
            in_channels_edge = input_dict['in_channels_edge'],
            hidden_channels = input_dict['hidden_channels'],
            out_channels = input_dict['out_channels'], 
            n_mlp_encode = input_dict['n_mlp_encode'], 
            n_mlp_mp = input_dict['n_mlp_mp'],
            n_mp_down_topk = input_dict['n_mp_down_topk'],
            n_mp_up_topk = input_dict['n_mp_up_topk'],
            pool_ratios = input_dict['pool_ratios'], 
            n_mp_down_enc = input_dict['n_mp_down_enc'], 
            n_mp_up_enc = input_dict['n_mp_up_enc'], 
            n_mp_down_dec = input_dict['n_mp_down_dec'], 
            n_mp_up_dec = input_dict['n_mp_up_dec'], 
            lengthscales_enc = input_dict['lengthscales_enc'],
            lengthscales_dec = input_dict['lengthscales_dec'], 
            bounding_box = input_dict['bounding_box'], 
            interpolation_mode = input_dict['interp'], 
            act = input_dict['act'], 
            param_sharing = input_dict['param_sharing'],
            filter_lengthscale = input_dict['filter_lengthscale'], 
            name = input_dict['name'])
    model_topk.load_state_dict(p['state_dict'])
    model_topk.to(device)
    model_topk.eval()

    # ~~~~ Re-load data: 
    rollout_eval = 1 # where to evaluate the RMSE  
    use_radius = True
    #vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_32564.vtk'
    vtk_file_test = 'datasets/BACKWARD_FACING_STEP/Backward_Facing_Step_Cropped_Re_26214.vtk'
    path_to_ei = 'datasets/BACKWARD_FACING_STEP/edge_index'
    path_to_ea = 'datasets/BACKWARD_FACING_STEP/edge_attr'
    path_to_pos = 'datasets/BACKWARD_FACING_STEP/pos'
    device_for_loading = device

    dataset_eval, _ = bfs.get_pygeom_dataset_cell_data(
                    vtk_file_test, 
                    path_to_ei, 
                    path_to_ea,
                    path_to_pos, 
                    device_for_loading, 
                    use_radius,
                    time_lag = rollout_eval,
                    scaling = [data_mean, data_std],
                    features_to_keep = [1,2], 
                    fraction_valid = 0, 
                    multiple_cases = False)

    # Loss 
    loss_fn = nn.MSELoss()



    # Get a prediction and its target  
    data = dataset_eval[0]
    
    rollout_length = rollout_eval 
    loss = torch.tensor([0.0])
    loss_scale = torch.tensor([1.0/rollout_length])

    # Rollout prediction: 
    x_new = data.x
    for t in range(rollout_length):
        x_old = torch.clone(x_new)
        x_src, mask = model_topk(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
        x_new = x_old + x_src

        # Evaluate the non-mask regularization
        non_mask = 1 - mask     
        non_mask = non_mask[:, None]
        lam = 1.0

        # Accumulate loss 
        target = data.y[t]
        loss += loss_scale * ( loss_fn(x_new, target) + lam*loss_fn(non_mask*x_new, non_mask*target) )





