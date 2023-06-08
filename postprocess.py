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

test_dataset, _ = bfs.get_pygeom_dataset_cell_data_radius(
                vtk_file_test, 
                path_to_ei, 
                path_to_ea,
                path_to_pos, 
                device_for_loading, 
                time_lag = 2,
                scaling = [data_mean, data_std],
                features_to_keep = [1,2], 
                fraction_valid = 0, 
                multiple_cases = False)

# Remove first snapshot
test_dataset.pop(0)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Postprocess losses 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:  
    # Load model 
    a = torch.load('saved_models/model_single_scale.tar')
    b = torch.load('saved_models/model_multi_scale.tar')
    #c = torch.load('saved_models/model_multi_scale_topk.tar.old')
    #c = torch.load('saved_models/topk_down_topk_1_1_up_topk_1_1_factor_16_hc_128_down_enc_4_up_enc_down_dec_4_4_4_up_dec_4_4_param_sharing_0.tar')
    c = torch.load('saved_models/topk_unet_down_topk_1_1_up_topk_1_factor_4_hc_128_down_enc_4_4_4_up_enc_4_4_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar')
    

    # Plot losses:
    fig, ax = plt.subplots(1,3,sharey=True, sharex=True)
    ax[0].plot(a['loss_hist_train'])
    ax[0].plot(a['loss_hist_test'])
    ax[0].set_yscale('log')
    #ax[0].set_ylim([1e-3, 1e-1])
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_title('Single Scale')

    ax[1].plot(b['loss_hist_train'], label='train')
    ax[1].plot(b['loss_hist_test'], label='valid')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Epochs')
    ax[1].set_title('Multi Scale')

    ax[2].plot(c['loss_hist_train'])
    ax[2].plot(c['loss_hist_test'])
    ax[2].set_yscale('log')
    ax[2].set_xlim([0,150])
    ax[2].set_xlabel('Epochs')
    ax[2].set_title('Multi Scale + Top-K')

    plt.show(block=False)



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load models and Plot losses 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 0:
    modelpath = 'saved_models/topk_down_topk_1_1_up_topk_1_1_factor_16_hc_128_down_enc_4_up_enc_down_dec_4_4_4_up_dec_4_4_param_sharing_0.tar'
    #modelpath = 'saved_models/model_multi_scale.tar'
    #modelpath = 'saved_models/model_single_scale.tar'
    p = torch.load(modelpath)

    input_dict = p['input_dict']
    print('input_dict: ', input_dict)
    p = torch.load(modelpath)

    input_dict = p['input_dict']
    print('input_dict: ', input_dict)

    
    # With top-k:
    model = gnn.GNN_TopK(
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


    # # Without top-k: 
    # model = gnn.Multiscale_MessagePassing_UNet(
    #             in_channels_node = input_dict['in_channels_node'],
    #             in_channels_edge = input_dict['in_channels_edge'],
    #             hidden_channels = input_dict['hidden_channels'],
    #             n_mlp_encode = input_dict['n_mlp_encode'],
    #             n_mlp_mp = input_dict['n_mlp_mp'],
    #             n_mp_down = input_dict['n_mp_down'],
    #             n_mp_up = input_dict['n_mp_up'],
    #             n_repeat_mp_up = input_dict['n_repeat_mp_up'],
    #             lengthscales = input_dict['lengthscales'],
    #             bounding_box = input_dict['bounding_box'],
    #             act = input_dict['act'],
    #             interpolation_mode = input_dict['interpolation_mode'],
    #             name = input_dict['name'])


    model.load_state_dict(p['state_dict'])
    model.eval()

    fig, ax = plt.subplots()
    colors = ['black','blue','red']
    ax.plot(p['loss_hist_train'], label='train', lw=2, color='black')
    ax.plot(p['loss_hist_test'], label='valid', lw=2, color='black', ls='--')

    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.grid(True)
    #ax.set_ylim([1e-6, 1])
    ax.set_ylim([1e-3, 0.1]) 

    ax.legend(fancybox=False, framealpha=1, edgecolor='black')
    ax.set_title('Training History')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    plt.show(block=False)





if 1 == 0:
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
# Test models 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if 1 == 1:

    # step 1: create model w/ shared params 
    bounding_box = test_dataset[0].bounding_box
    bbox = [tnsr.item() for tnsr in bounding_box]
    model = gnn.GNN_TopK_NoReduction(
                in_channels_node = 2,
                in_channels_edge = 3,
                hidden_channels = 128,
                out_channels = 2,
                n_mlp_encode = 3,
                n_mlp_mp = 2,
                n_mp_down_topk = [2],
                n_mp_up_topk = [],
                pool_ratios = [1./4.],
                n_mp_down_enc = [4,4,4],
                n_mp_up_enc = [4,4],
                n_mp_down_dec = [4,4,4],
                n_mp_up_dec = [4,4],
                lengthscales_enc = [0.01, 0.02],
                lengthscales_dec = [0.01, 0.02],
                bounding_box = bbox,
                interpolation_mode = 'knn',
                act = F.elu,
                param_sharing = True,
                name = 'topk_unet')



    # Do a forward pass 
    data = test_dataset[0]
    print('Exec forward pass..')
    x_src = model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)


    sd = model.state_dict()


    # step 2: create model with unshared params 
    model_2 = gnn.GNN_TopK_NoReduction(
                in_channels_node = 2,
                in_channels_edge = 3,
                hidden_channels = 128,
                out_channels = 2,
                n_mlp_encode = 3,
                n_mlp_mp = 2,
                n_mp_down_topk = [1,1],
                n_mp_up_topk = [1],
                pool_ratios = [1./4.],
                n_mp_down_enc = [4,4,4],
                n_mp_up_enc = [4,4],
                n_mp_down_dec = [4,4,4],
                n_mp_up_dec = [4,4],
                lengthscales_enc = [0.01, 0.02],
                lengthscales_dec = [0.01, 0.02],
                bounding_box = bbox,
                interpolation_mode = 'knn',
                act = F.elu,
                param_sharing = False,
                name = 'topk_unet_2')

    #model_2.set_down_mmp_block(model.down_mps, 0)
        
    print('overwriting down_mps block')
    model_2.set_mmp_layer(model.down_mps, 0, block_to_write='down')
    print('overwriting up_mps block')
    model_2.set_mmp_layer(model.down_mps, 0, block_to_write='up')

    
    print('node/edge encode')
    model_2.set_node_edge_encoder_decoder(model)

    # node encode norm
    
    # edge encode norm 

    # node decode 







# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Fixing parameters - isolating impact of top-k 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~







