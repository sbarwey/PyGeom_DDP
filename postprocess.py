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

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def ratio_boundary_internal_nodes(p: int) -> float: 
    ratio = ( (p+1)**3 - (p-1)**3 )/(p-1)**3
    return ratio 

if __name__ == "__main__":

    
    # ~~~~ postprocessing: training losses 
    if 1 == 0:
        mp = [1,2,3,4,5,6,7,8]
        mp = [6]

        conv_loss_train = []
        conv_loss_valid = []
 
        plt.rcParams.update({'font.size': 18})
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        for i in mp:
            a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(i))
            ax.plot(a['loss_hist_train'], label='mp %d' %(i), lw=2, color='red')
            ax2.plot(a['lr_hist'],  label='mp %d' %(i), lw=2)
            conv_loss_train.append(np.mean(a['loss_hist_train'][-10:]))
            conv_loss_valid.append(np.mean(a['loss_hist_test'][-10:]))
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(False)
        plt.show(block=False)
     
        # ms=15
        # mew=2
        # lw=1
        # fig, ax = plt.subplots()
        # ax.plot(mp, conv_loss_train, marker='o', fillstyle='none', lw=lw, ms=ms, mew=mew, color='black')
        # ax.plot(mp, conv_loss_valid, marker='s', fillstyle='none', lw=lw, ms=ms, mew=mew, color='blue')
        # ax.set_ylabel('Super-Resolution Loss')
        # ax.set_xlabel('Message Passing Layers')
        # #ax.set_ylim([0.05, 0.07])
        # #ax.set_yscale('log')
        # plt.show(block=False)

        mp = 6 
        a = torch.load('./saved_models/single_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mp))
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mp))
        epochs = list(range(1, 300))
        plt.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots()
        ax.plot(epochs, a['loss_hist_train'][1:], lw=2, color='red', label='Single-Scale')
        ax.plot(epochs, a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(epochs, b['loss_hist_train'][1:], lw=2, color='black', label='Multi-Scale')
        ax.plot(epochs, b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        
        # ax2 = ax.twinx()
        # ax2.plot(a['lr_hist'][1:],  label='mp %d' %(i), lw=2, color='blue')
        # ax2.set_ylabel('Learning Rate', color='blue')
        # ax2.tick_params(axis='y', labelcolor='blue')
        # ax2.grid(False)

        plt.show(block=False)



    # ~~~~ Analyze predictions -- all elements 
    if 1 == 0:
        mode = "multi_scale"

        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)

        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        
        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mode,mp))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model.load_state_dict(a['state_dict'])
        device = 'cpu'
        model.to(device)
        model.eval()

        with torch.no_grad():
            x_input_list = [] 
            y_pred_list = []
            y_target_list = []

            dataset = test_dataset 
            N = len(dataset) 
            mse_before_gnn = torch.zeros(N)
            mse_after_gnn = torch.zeros(N) 

            for i in range(N):
                print('evaluating %d/%d' %(i+1, N))
                data = dataset[i]

                # 1) Preprocessing: scale input  
                eps = 1e-10
                x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                # 2) evaluate model 
                out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                    
                # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                y_pred = out_gnn * (data.x_std + eps) + data.x 

                # y_pred_list.append(y_pred)
                # y_target_list.append(y_target)
                # x_input_list.append(x_input)

                # loss before gnn 
                mse_before_gnn[i] = F.mse_loss(data.x, data.y)

                # loss after gnn 
                mse_after_gnn[i] = F.mse_loss(y_pred, data.y) 
        asdf

        # Visualize
        element_ids = torch.arange(N) 
    
        # Get the difference in MSE 
        # if negative, bad
        # if positive, good 
        difference = mse_before_gnn - mse_after_gnn

        # sort the difference  
        idx_sort = torch.sort(difference, descending=True)[1]

        element_ids_sorted = element_ids[idx_sort]
        difference_sorted = difference[idx_sort]

        plt.rcParams.update({'font.size': 14})

        # Plotting error difference versus element 
        fig, ax = plt.subplots()
        ax.plot(element_ids, difference_sorted, lw=2)
        #ax.plot(element_ids, difference_sorted.abs(), lw=2)
        ax.set_xlabel('Element IDs (Sorted)')
        ax.set_ylabel('Model Gain')
        #ax.set_yscale('log')
        plt.show(block=False)

        # # Plotting error difference versus RMS 
        # fig, ax = plt.subplots()
        # ax.scatter(difference, rms_target)
        # ax.set_xlabel('Model Gain')
        # ax.set_ylabel('Target RMS Velocity')
        # plt.show(block=False)

    # ~~~~ Visualize model predictions -- a single element  
    if 1 == 1:
        mode = "single_scale"

        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)

        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        
        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_%d.tar' %(mode,mp))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
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
            # eids = torch.load('./outputs/postproc/element_ids_sorted.pt')
            # data = test_dataset[ eids[0] ]
            data = test_dataset[150]
             
            # 1) Preprocessing: scale input  
            eps = 1e-10
            x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

            # 2) evaluate model 
            out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                
            # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
            y_pred = out_gnn * (data.x_std + eps) + data.x 

        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        for comp in range(3):
            ax[comp].scatter(data.x[:,comp], data.y[:,comp], color='red')
            ax[comp].scatter(y_pred[:,comp], data.y[:,comp], color='lime')
            ax[comp].plot([data.y[:,comp].min(), data.y[:,comp].max()],
                    [data.y[:,comp].min(), data.y[:,comp].max()],
                    color='black', lw=2)
            ax[comp].set_title('n_mp=%d, comp=%d' %(mp,comp))
            ax[comp].set_xlabel('Prediction')
            ax[comp].set_ylabel('Target')
        plt.show(block=False)

    # ~~~~ Postprocess run logs: KE, dissipation, enst
    if 1 == 0:
        import re

        def read_nrs_log(file_path):
            lines = []
            values = []
            with open(file_path, 'r') as file:
                 for line in file:
                     if line.startswith('  time'):
                         line=line.strip()
                         lines.append(line)
                         numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                         numbers = [float(num) for num in numbers]
                         values.append(np.array(numbers, ndmin=2))
            del lines[0]
            del values[0]
            values = np.concatenate(values, axis=0) 
            return lines, values 

        # Re = 1600 
        _, values_1600 = read_nrs_log('./outputs/run_logs/36_cubed/Re_1600_poly_7.log')
        _, values_2000 = read_nrs_log('./outputs/run_logs/36_cubed/Re_2000_poly_7.log')
        _, values_2400 = read_nrs_log('./outputs/run_logs/36_cubed/Re_2400_poly_7.log')

        _, values_1600_2 = read_nrs_log('./outputs/run_logs/72_cubed/Re_1600_poly_7.log')
        
        #   0      1     2          4        5       6
        # [time, enst, energy, -2*nu*enst, dE/dt, nuEff/nu]

        # Kinetic energy  
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], values_1600[:,2], label='Re=1600, 36^3, P=7', lw=lw)
        ax.plot(values_1600_2[:,0], values_1600_2[:,2], label='Re=1600, 72^3, P=7', lw=lw)
        # ax.plot(values_2000[:,0], values_2000[:,2], label='Re=2000', lw=lw)
        # ax.plot(values_2400[:,0], values_2400[:,2], label='Re=2400', lw=lw)
        ax.set_title('Kinetic Energy')
        #ax.set_xlim([7.5, 10.5])
        plt.show(block=False)

        # dissipation rate ( dEk/dt )
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], -values_1600[:,4], label='Re=1600, 36^3, P=7', lw=lw)
        ax.plot(values_1600_2[:,0], -values_1600_2[:,4], label='Re=1600, 72^3, P=7', lw=lw, ls='--')
        #ax.plot(values_2000[:,0], -values_2000[:,5], label='Re=2000', lw=lw)
        #ax.plot(values_2400[:,0], -values_2400[:,5], label='Re=2400', lw=lw)
        ax.set_title('Dissipation Rate')
        #ax.set_xlim([7.5, 10.5])
        ax.legend()
        ax.set_xlabel('t_c')
        #ax.set_ylabel('-dEk/dt')
        ax.set_ylabel('-dEk/dt')
        plt.show(block=False)

    # ~~~~ Plotting  
    if 1 == 0:
        print('Plotting...')

        # Plot the edges 
        if 1 == 0:
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
    
        # Plot scatter plots of node features 
        if 1 == 0:
            e_id = 0

            sample = train_dataset[e_id]

            color='red' 
            fig, ax = plt.subplots(1,3, figsize=(10,4))
            
            for comp in range(3):
                ax[comp].scatter(sample.x[:,comp], sample.y[:,comp], c=color)
                ax[comp].plot( [sample.y[:,comp].min(), sample.y[:,comp].max()], 
                               [sample.y[:,comp].min(), sample.y[:,comp].max()],
                           color='black', lw=2)

            ax[0].set_xlabel('x')
            ax[0].set_ylabel('y')
            ax[0].set_title('e_id = %d' %(e_id))
            ax[0].set_aspect('equal')

            ax[1].set_xlabel('x')
            ax[1].set_ylabel('y')
            ax[1].set_title('e_id = %d' %(e_id))
            ax[1].set_aspect('equal')

            ax[2].set_xlabel('x')
            ax[2].set_ylabel('y')
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

    # ~~~~ Test the model 
    if 1 == 0:   
        sample = train_dataset[0]
        input_node_channels = sample.x.shape[1]
        input_edge_channels = sample.pos.shape[1] + sample.x.shape[1] + 1 
        hidden_channels = 128  
        output_node_channels = sample.y.shape[1] 
        n_mlp_hidden_layers = 2 
        n_messagePassing_layers = 8

        name = 'gnn' 
        model = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)

        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=False,
        )

        for bidx, data in enumerate(train_loader):
            # 1) Preprocessing: scale input  
            eps = 1e-10
            x_scaled = (data.x - data.x_mean)/(data.x_std + eps) 

            # 2) Evaluate the model 
            gnn_out = model(x_scaled, data.edge_index, data.pos_norm, data.batch)

            # 3) Set the target 
            target = (data.y - data.x)/(data.x_std + eps)

            # 4) Inference -- Make a prediction 
            # y_pred = gnn_out * (data.x_std.unsqueeze(0) + eps) + data.x 

            break





