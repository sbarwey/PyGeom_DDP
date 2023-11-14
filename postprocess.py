import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
import matplotlib.pyplot as plt
import dataprep.nekrs_graph_setup as ngs
import models.gnn as gnn

seed = 122
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_grad_enabled(False)

def ratio_boundary_internal_nodes(p: int) -> float: 
    ratio = ( (p+1)**3 - (p-1)**3 )/(p-1)**3
    return ratio 

if __name__ == "__main__":

    case_path = './datasets/tgv'
    device_for_loading = 'cpu'
    fraction_valid = 0.1 

    train_dataset = []
    test_dataset = []  

    data_read_world_size = 4 

    for i in range(data_read_world_size):
        if i != 3:
            continue
        data_x_path = case_path + '/gnn_outputs_original_poly_7' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # input  
        data_y_path = case_path + '/gnn_outputs_recon_poly_7' + '/fld_u_rank_%d_size_%d' %(i,data_read_world_size) # target 
        edge_index_path = case_path + '/gnn_outputs_original_poly_7' + '/edge_index_element_local_rank_%d_size_%d' %(i,data_read_world_size) 
        node_element_ids_path = case_path + '/gnn_outputs_original_poly_7' + '/node_element_ids_rank_%d_size_%d' %(i,data_read_world_size)
        global_ids_path = case_path + '/gnn_outputs_original_poly_7' + '/global_ids_rank_%d_size_%d' %(i,data_read_world_size) 
        pos_path = case_path + '/gnn_outputs_original_poly_7' + '/pos_node_rank_%d_size_%d' %(i,data_read_world_size) 

        # note: data_mean and data_std are overwritten. the one used is for i = 3 
        train_dataset_temp, test_dataset_temp, data_mean, data_std = ngs.get_pygeom_dataset(
                                                             data_x_path, 
                                                             data_y_path,
                                                             edge_index_path,
                                                             node_element_ids_path,
                                                             global_ids_path,
                                                             pos_path,
                                                             device_for_loading,
                                                             fraction_valid)

        train_dataset += train_dataset_temp
        test_dataset += test_dataset_temp

    # ~~~~ postprocessing: training losses 
    if 1 == 0:
        mp = [1,2,3,4,5,6,7,8]

        conv_loss_train = []
        conv_loss_valid = []
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        for i in mp:
            a = torch.load('./saved_models/super_res_gnn_mp_%d.tar' %(i))
            ax.plot(a['loss_hist_train'], label='mp %d' %(i))
            conv_loss_train.append(np.mean(a['loss_hist_train'][-10:]))
            conv_loss_valid.append(np.mean(a['loss_hist_test'][-10:]))
        ax.set_yscale('log')
        ax.legend()
        plt.show(block=False)
     
        ms=15
        mew=2
        lw=1
        fig, ax = plt.subplots()
        ax.plot(mp, conv_loss_train, marker='o', fillstyle='none', lw=lw, ms=ms, mew=mew, color='black')
        ax.plot(mp, conv_loss_valid, marker='s', fillstyle='none', lw=lw, ms=ms, mew=mew, color='blue')
        ax.set_ylabel('Super-Resolution Loss')
        ax.set_xlabel('Message Passing Layers')
        ax.set_yscale('log')
        plt.show(block=False)


    # ~~~~ Visualize model predictions 
    if 1 == 1:
        mp = [1,2,3,4,5,6,7,8] 
        x_list = [] 
        y_pred_list = []
        y_target_list = []

        for i in mp:
            a = torch.load('./saved_models/super_res_gnn_mp_%d.tar' %(i))
            sample = train_dataset[0]
            n_messagePassing_layers = i 
            
            input_dict = a['input_dict']
            input_channels = input_dict['input_channels']
            hidden_channels = input_dict['hidden_channels']
            output_channels = input_dict['output_channels']
            n_mlp_layers = input_dict['n_mlp_layers']
            activation = input_dict['activation']
            name = 'temp'

            model = gnn.mp_gnn(input_channels,
                           hidden_channels,
                           output_channels,
                           n_mlp_layers,
                           n_messagePassing_layers,
                           activation,
                           name)
            
            model.load_state_dict(a['state_dict'])
            device = 'cpu'
            model.to(device)
            model.eval()

            with torch.no_grad():

                #element 1: 100 
                #element 2: 1000
                #element 3: 23 
                #element 4: 1100
                data = test_dataset[1100]

                # scale input 
                x_in = (data.x - data_mean[0])/data_std[0]

                # evaluate 
                y_pred_scaled = model(x_in, data.edge_index, data.pos_norm, data.batch)

                # unscale output 
                y_pred = y_pred_scaled * data_std[1] + data_mean[1] 
                y_target = data.y
                x = data.x

                y_pred_list.append(y_pred)
                y_target_list.append(y_target)
                x_list.append(x)


        plt.rcParams.update({'font.size': 14})
        for comp in range(3):
            fig, ax = plt.subplots(2,4, figsize=(15,7))
            count = 0 
            for r in range(2):
                for c in range(4):
                    i = mp[count]
                    x = x_list[count]
                    y_target = y_target_list[count]
                    y_pred = y_pred_list[count]
                    ax[r,c].scatter(x[:,comp], y_target[:,comp], color='red')
                    ax[r,c].scatter(y_pred[:,comp], y_target[:,comp], color='lime')
                    ax[r,c].plot([y_target[:,comp].min(), y_target[:,comp].max()],
                               [y_target[:,comp].min(), y_target[:,comp].max()],
                               color='black', lw=2)
                    ax[r,c].set_title('n_mp=%d' %(i))
                    ax[r,c].set_xlabel('Prediction')
                    ax[r,c].set_ylabel('Target')
                    count += 1 
            plt.show(block=False)

    # ~~~~ Plotting  
    if 1 == 1:
        print('Plotting...')

        # Plot the edges 
        if 1 == 1:
            #for rl in range(SIZE):
            data = train_dataset[0]
            G = utils.to_networkx(data=data)

            
            pos = dict(enumerate(np.array(data.pos)))
            node_xyz = np.array([pos[v] for v in sorted(G)])
            edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
            pos = data.pos

            ms = 50
            lw_edge = 2
            lw_marker = 1
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the edges
            count = 0
            for vizedge in edge_xyz:
                #ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.1)
                ax.plot(*vizedge.T, color="black", alpha=0.3)
                count += 1

            # plot the nodes 
            ax.scatter(*pos.T, s=ms, ec='black', lw=lw_marker, c='black', alpha=1)

            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=34, azim=-24, roll=0)
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.show(block=False)

        # Plot fine and coarse element positions, for one element   
        if 1 == 0:
            e_id = 510
            ms = 50
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos_hi[e_id][:,0], pos_hi[e_id][:,1], pos_hi[e_id][:,2], c='black', marker='o', s=ms)
            ax.scatter(pos_lo[e_id][:,0], pos_lo[e_id][:,1], pos_lo[e_id][:,2], c='blue', marker='o', s=ms)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show(block=False)

        # Isolate internal nodes from boundary nodes 
        if 1 == 0: 
            e_id = 325
            mask_internal = gli_lo[e_id] == 0
            mask_boundary = ~mask_internal 
            ms = 50
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pos_hi[e_id][mask_internal,0], 
                       pos_hi[e_id][mask_internal,1], 
                       pos_hi[e_id][mask_internal,2], c='blue', marker='o', s=ms)
            ax.scatter(pos_hi[e_id][mask_boundary,0], 
                       pos_hi[e_id][mask_boundary,1], 
                       pos_hi[e_id][mask_boundary,2], c='red', marker='o', s=ms)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show(block=False)
    
        # Plot scatter plots of node features 
        if 1 == 0:
            e_id = 45

            color='red' 
            fig, ax = plt.subplots(1,3, figsize=(10,4))
            ax[0].scatter(x_lo[e_id][:,0], x_hi[e_id][:,0], c=color)
            ax[0].plot([x_hi[e_id][:,0].min(), x_hi[e_id][:,0].max()], 
                       [x_hi[e_id][:,0].min(), x_hi[e_id][:,0].max()],
                       color='black', lw=2)

            ax[1].scatter(x_lo[e_id][:,1], x_hi[e_id][:,1], c=color)
            ax[1].plot([x_hi[e_id][:,1].min(), x_hi[e_id][:,1].max()], 
                       [x_hi[e_id][:,1].min(), x_hi[e_id][:,1].max()],
                       color='black', lw=2)

            ax[2].scatter(x_lo[e_id][:,2], x_hi[e_id][:,2], c=color)
            ax[2].plot([x_hi[e_id][:,2].min(), x_hi[e_id][:,2].max()], 
                       [x_hi[e_id][:,2].min(), x_hi[e_id][:,2].max()],
                       color='black', lw=2)
            
            ax[0].set_ylabel('u_f')
            ax[0].set_xlabel('P u_c')
            ax[0].set_title('e_id = %d' %(e_id))
            ax[0].set_aspect('equal')

            ax[1].set_ylabel('u_f')
            ax[1].set_xlabel('P u_c')
            ax[1].set_title('e_id = %d' %(e_id))
            ax[1].set_aspect('equal')

            ax[2].set_ylabel('u_f')
            ax[2].set_xlabel('P u_c')
            ax[2].set_title('e_id = %d' %(e_id))
            ax[2].set_aspect('equal')
            plt.show(block=False)

        # RMS versus reconstruction MSE 
        if 1 == 0:
            fig, ax = plt.subplots(figsize=(8,7))
            ax.scatter(rms_lo, element_error, facecolor='none', edgecolor='black', lw=0.75, s=75, label='Target')
            ax.scatter(rms_hi, element_error, facecolor='none', edgecolor='red', lw=0.75, s=75, label='Predicted')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('RMS Velocity', fontsize=20)
            ax.set_ylabel('Reconstruction MSE', fontsize=20)
            ax.legend(fontsize=20)
            plt.show(block=False)


        # Error budget: boundary vs internal 
        if 1 == 0:
            # get prediction errors 
            element_error = (x_hi - x_lo)**2
            element_error = element_error.mean(dim=[1,2])

            # contribution of boundary nodes to the error 
            mask_internal = gli_lo == 0
            mask_boundary = ~mask_internal
             
            mask_internal = mask_internal.unsqueeze(-1)
            mask_boundary = mask_boundary.unsqueeze(-1)

            element_error_boundary =  ((mask_boundary*x_hi - mask_boundary*x_lo)**2).mean(dim=[1,2]) 
            element_error_internal =  ((mask_internal*x_hi - mask_internal*x_lo)**2).mean(dim=[1,2]) 
           
            # sort elements by error 
            error_total, idx_sort = torch.sort(element_error, descending=True)

            error_boundary = element_error_boundary[idx_sort]
            error_internal = element_error_internal[idx_sort]

            lw=1
            fig, ax = plt.subplots(1,2, figsize=(10,4))
            ax[0].plot(error_total, color='black', lw=lw, label='Total Error')
            ax[0].plot(error_boundary, color='red', lw=lw, label='From Boundary Nodes')
            ax[0].plot(error_internal, color='blue', lw=lw, label='From Internal Nodes')
            ax[0].set_xlabel('Element ID')
            ax[0].set_ylabel('Reconstruction Error')

            ax[1].plot((error_boundary/error_total)*100, color='red', lw=lw, label='From Boundary Nodes')
            ax[1].plot((error_internal/error_total)*100, color='blue', lw=lw, label='From Internal Nodes')
            ax[1].set_xlabel('Element ID')
            ax[1].set_ylabel('Error Contribution [%]')

            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax[0].legend(fancybox=False, prop={'size': 14})
            plt.show(block=False)
        



