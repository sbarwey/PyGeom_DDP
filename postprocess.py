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
from pymech.neksuite import readnek,writenek
from pymech.dataset import open_dataset
from typing import Optional, Union, Callable, List, Tuple


seed = 122
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_grad_enabled(False)

def count_parameters(mdl):
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)

def get_edge_index(edge_index_path: str,
                   edge_index_vertex_path: Optional[str] = None) -> torch.Tensor:
    print('Loading edge index')
    edge_index = np.loadtxt(edge_index_path, dtype=np.int64).T
    if edge_index_vertex_path:
        print('Adding p1 connectivity...')
        print('\tEdge index shape before: ', edge_index.shape)
        edge_index_vertex = np.loadtxt(edge_index_vertex_path, dtype=np.int64).T
        edge_index = np.concatenate((edge_index, edge_index_vertex), axis=1)
        print('\tEdge index shape after: ', edge_index.shape)
    edge_index = torch.tensor(edge_index)
    return edge_index 

if __name__ == "__main__":
    # ~~~~ Spectrum plots 
    if 1 == 1:
        
        # nekrs interp : 
        t_snap = "16"
        data_nrs_1to7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_interp_1to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_knn_1to7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_knninterp_1to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_tgt_7 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_target/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_crs_7to1 = np.load(f"./outputs/Re_1600_poly_7_testset/one_shot/snapshots_coarse_7to1/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")

        # incr 
        data_nrs_5to7 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_interp_full_5to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_knn_5to7 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_knninterp_5to7/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")
        data_crs_3to1 = np.load(f"./outputs/Re_1600_poly_7_testset/incr/snapshots_coarse_3to1/regtgv_reg0.f000{t_snap}-SPECTRUM.npz")

        # knn interp :
        lw = 2
        fig, ax = plt.subplots()
        ax.plot(data_tgt_7['kspec'], data_tgt_7['spectrum'], color='black', lw=lw, label='Target')
        ax.plot(data_crs_7to1['kspec'], data_crs_7to1['spectrum'], color='black', lw=lw, ls='--', label='P=1')
        #ax.plot(data_crs_3to1['kspec'], data_crs_3to1['spectrum'], color='gray', lw=lw, ls='-.', label='P=1 (from 3)')
        ax.plot(data_nrs_1to7['kspec'], data_nrs_1to7['spectrum'], color='blue', lw=lw, ls='--', label='NekRS')
        #ax.plot(data_nrs_5to7['kspec'], data_nrs_5to7['spectrum'], color='cyan', lw=lw, ls='--', label='NekRS-incr')
        #ax.plot(data_knn_1to7['kspec'], data_knn_1to7['spectrum'], color='red', lw=lw, ls='--', label='kNN')
        #ax.plot(data_knn_5to7['kspec'], data_knn_5to7['spectrum'], color='magenta', lw=lw, ls='--', label='kNN-incr')

        # plot vlines: p = 1 
        ax.vlines(data_crs_7to1['nyq_size'],  1e-9, 1e-1, lw=lw, color='gray', zorder=-1)
        ax.vlines(data_tgt_7['nyq_size'],  1e-9, 1e-1, lw=lw, color='gray', zorder=-1)

        plt.show(block=False)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim([1,300])
        ax.set_ylim([1e-9, 1e-1])
        ax.set_ylabel('E(k)')
        ax.set_xlabel('k')
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.4)
        ax.legend(fancybox=False, framealpha=1, edgecolor='black')
        plt.show(block=False)
            
        pass




    # ~~~~ postprocessing: training losses -- comparing a set of different models 
    if 1 == 0:
        n_mp = 12
        fine_mp = 'False'

        # one-shot -- single-scale 
        a = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        a_label = '1shot+0nei'
        a_color = 'black'
        a_ls = '-'

        # incremental - singlescale 
        b = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        b_label = '1shot+6nei'
        b_color = 'blue'
        b_ls = '-'

        # fine-scale neighbors -- single-scale 
        c = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        c_label = '1shot+26nei'
        c_color = 'red'
        c_ls = '-'

        # coarse-scale neighbors -- single-scale 
        d = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_0_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        d_label = '1shot+0nei, Resid'
        d_color = 'lime'
        d_ls = '-'

        # coarse-scale neighbors -- single-scale 
        e = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_6_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        e_label = '1shot+6nei, Resid'
        e_color = 'gray'
        e_ls = '-'
    
        # coarse-scale neighbors -- single-scale 
        f = torch.load(f'./saved_models/single_scale/gnn_lr_1em4_bs_4_nei_26_c2f_multisnap_resid_3_7_132_128_3_2_{n_mp}_{fine_mp}.tar')
        f_label = '1shot+26nei, Resid'
        f_color = 'magenta'
        f_ls = '-'

        plt.rcParams.update({'font.size': 18})
        
        fig, ax = plt.subplots(figsize=(8,6))

        ax.plot(a['loss_hist_train'][0:], lw=2, color=a_color, label=a_label, ls=a_ls)
        ax.plot(b['loss_hist_train'][0:], lw=2, color=b_color, label=b_label, ls=b_ls)
        ax.plot(c['loss_hist_train'][0:], lw=2, color=c_color, label=c_label, ls=c_ls)
        ax.plot(d['loss_hist_train'][0:], lw=2, color=d_color, label=d_label, ls=d_ls)
        ax.plot(e['loss_hist_train'][0:], lw=2, color=e_color, label=e_label, ls=e_ls)
        ax.plot(f['loss_hist_train'][0:], lw=2, color=f_color, label=f_label, ls=f_ls)

        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(n_mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        ax.set_ylim([1e-4, 1e0])
        
        plt.show(block=False)

    # ~~~~ postprocessing: training losses -- comparing effect of batch size  
    if 1 == 0:
        mp = 6 

        # one-shot -- multiscale 
        a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_3_7_128_3_2_6.tar')
        a_label = '8x32'
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_64_multisnap_3_7_128_3_2_6.tar')
        b_label = '8x64'
        c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_128_multisnap_3_7_128_3_2_6.tar')
        c_label = '8x128'

        # # incremental - singlescale 
        # a = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # a_label = '8x32'
        # b = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_64_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # b_label = '8x64'
        # c = torch.load('./saved_models/single_scale/gnn_lr_1em4_bs_128_multisnap_incr_v2_3_7_128_3_2_6.tar')
        # c_label = '8x128'

        epochs = list(range(1, 300))
        plt.rcParams.update({'font.size': 18})

        fig, ax = plt.subplots()
        ax.plot(a['loss_hist_train'][1:], lw=2, color='red', label=a_label)
        #ax.plot(a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(b['loss_hist_train'][1:], lw=2, color='black', label=b_label)
        #ax.plot(b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.plot(c['loss_hist_train'][1:], lw=2, color='blue', label=c_label)
        #ax.plot(c['loss_hist_test'][:-1], lw=2, color='blue', ls='--')

        # incremental - multiscale
        a = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_6.tar')
        a_label = '8x32'
        b = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_64_multisnap_incr_v2_3_7_128_3_2_6.tar')
        b_label = '8x64'
        c = torch.load('./saved_models/multi_scale/gnn_lr_1em4_bs_128_multisnap_incr_v2_3_7_128_3_2_6.tar')
        c_label = '8x128'

        ax.plot(a['loss_hist_train'][1:], lw=2, color='red', label=a_label, ls='--')
        #ax.plot(a['loss_hist_test'][:-1], lw=2, color='red', ls='--')
        ax.plot(b['loss_hist_train'][1:], lw=2, color='black', label=b_label, ls='--')
        #ax.plot(b['loss_hist_test'][:-1], lw=2, color='black', ls='--')
        ax.plot(c['loss_hist_train'][1:], lw=2, color='blue', label=c_label, ls='--')
        #ax.plot(c['loss_hist_test'][:-1], lw=2, color='blue', ls='--')

        ax.set_yscale('log')
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('n_mp_layers = %d' %(mp))
        #ax.tick_params(axis='y', labelcolor='red')  # Set y-axis tick labels to red
        
        # ax2 = ax.twinx()
        # ax2.plot(a['lr_hist'][1:],  label='mp %d' %(i), lw=2, color='blue')
        # ax2.set_ylabel('Learning Rate', color='blue')
        # ax2.tick_params(axis='y', labelcolor='blue')
        # ax2.grid(False)

        plt.show(block=False)

    # ~~~~ Save predicted flowfield into .f file 
    # ONE-SHOT and INCREMENTAL, WITHOUT NEIGHBORS
    if 1 == 0:
        mode = "single_scale"
        #data_dir = "./datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        #test_dataset = torch.load(data_dir + "/valid_dataset.pt")
        #edge_index = test_dataset[0].edge_index

        # Load model 
        mp = 6 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_bs_4_multisnap_3_7_128_3_2_%d.tar' %(mode,mp))
        #a = torch.load('./saved_models/%s/gnn_lr_1em4_bs_32_multisnap_incr_v2_3_7_128_3_2_%d.tar' %(mode,mp))
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
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model.to(device)
        model.eval()

        # Load eval and target snapshot 
        TORCH_FLOAT = torch.float32
        #nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
        #nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr'
        nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/one_shot'
        
        # Load in edge index 
        poly = 7
        case_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv"
        Re = '1600'
        edge_index_path = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly}/edge_index_element_local_rank_0_size_4"
        if mode == "single_scale":
            edge_index = get_edge_index(edge_index_path)
        elif mode == "multi_scale":
            edge_index_vertex_path = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly}/edge_index_element_local_vertex_rank_0_size_4"
            edge_index = get_edge_index(edge_index_path,
                                        edge_index_vertex_path)


        t_str_list = ['00017','00019', '00020','00021'] # 1 takes ~5 min 
        t_str_list = ['00017', '00020']
        #t_str_list = ['000%02d' %(i) for i in range(12,41)]

        for t_str in t_str_list:
            # Incremental: 
            #x_field = readnek(nrs_snap_dir + f'/snapshots_interp_1to3/newtgv0.f{t_str}')
            #x_field = readnek(nrs_snap_dir + f'/snapshots_interp_{poly-2}to{poly}/newtgv0.f{t_str}')
            #x_field = readnek(nrs_snap_dir + f'/snapshots_gnn_correction_{mode}_interp_{poly-2}to{poly}/newtgv0.f{t_str}')
            # One-shot
            x_field = readnek(nrs_snap_dir + f'/snapshots_interp_1to{poly}/newtgv0.f{t_str}')

            # y_field = readnek(nrs_snap_dir + f'/snapshots_target/tgv0.f{t_str}')
            n_snaps = len(x_field.elem)

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"Evaluating snap {i}/{n_snaps}")
                    pos_i = torch.tensor(x_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    vel_x_i = torch.tensor(x_field.elem[i].vel).reshape((3, -1)).T
                    # vel_y_i = torch.tensor(y_field.elem[i].vel).reshape((3, -1)).T

                    # get x_mean and x_std 
                    x_mean_element = torch.mean(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)
                    x_std_element = torch.std(vel_x_i, dim=0).unsqueeze(0).repeat(vel_x_i.shape[0], 1)

                    # element lengthscale 
                    lengthscale_element = torch.norm(pos_i.max(dim=0)[0] - pos_i.min(dim=0)[0], p=2)

                    # create data 
                    data = Data( x = vel_x_i.to(dtype=TORCH_FLOAT),
                                      # y = vel_y_i.to(dtype=TORCH_FLOAT),
                                      x_mean = x_mean_element.to(dtype=TORCH_FLOAT),
                                      x_std = x_std_element.to(dtype=TORCH_FLOAT),
                                      L = lengthscale_element.to(dtype=TORCH_FLOAT),
                                      pos = pos_i.to(dtype=TORCH_FLOAT),
                                      pos_norm = (pos_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                                      edge_index = edge_index)

                    data = data.to(device)

                    # ~~~~ Model evaluation ~~~~ # 
                    # 1) Preprocessing: scale input  
                    eps = 1e-10
                    x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                    # 2) evaluate model 
                    out_gnn = model(x_scaled, data.edge_index, data.pos_norm, data.batch)
                        
                    # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                    y_pred = out_gnn * (data.x_std + eps) + data.x 

                    # ~~~~ Making the .f file ~~~~ # 
                    # Re-shape the prediction, convert back to fp64 numpy 
                    y_pred = y_pred.cpu()
                    orig_shape = x_field.elem[i].vel.shape
                    y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()

                    # Place back in the snapshot data 
                    x_field.elem[i].vel[:,:,:,:] = y_pred_rs

                # Write 
                print('Writing...')
                # incremental: 
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_full_{poly-2}to{poly}"
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_{poly-2}to{poly}"
                # One shot: 
                directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_1to{poly}"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                    print(f"Directory '{directory_path}' created.")
                writenek(directory_path +  f"/newtgv0.f{t_str}", x_field)
                print(f'finished writing {t_str}') 



        # plt.rcParams.update({'font.size': 14})
        # fig, ax = plt.subplots(1,3, figsize=(10,4))
        # for comp in range(3):
        #     ax[comp].scatter(data.x[:,comp], data.y[:,comp], color='red')
        #     ax[comp].scatter(y_pred[:,comp], data.y[:,comp], color='lime')
        #     ax[comp].plot([data.y[:,comp].min(), data.y[:,comp].max()],
        #             [data.y[:,comp].min(), data.y[:,comp].max()],
        #             color='black', lw=2)
        #     ax[comp].set_title('n_mp=%d, comp=%d' %(mp,comp))
        #     ax[comp].set_xlabel('Prediction')
        #     ax[comp].set_ylabel('Target')
        # plt.show(block=False)

    # ~~~~ Save predicted flowfield into .f file 
    # COARSE-TO-FINE GNN 
    if 1 == 0:
        local = True
        mode = "single_scale"

        # Load model 
        mp = 6 
        n_element_neighbors = 26
        batch_size = 4
        a = torch.load(f"./saved_models/{mode}/gnn_lr_1em4_bs_{batch_size}_nei_{n_element_neighbors}_c2f_multisnap_3_7_132_128_3_2_{mp}.tar")

        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels_coarse = input_dict['input_edge_channels_coarse'] 
        input_edge_channels_fine = input_dict['input_edge_channels_fine'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']

        model = gnn.GNN_Element_Neighbor_Lo_Hi(
                           input_node_channels,
                           input_edge_channels_coarse,
                           input_edge_channels_fine,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)

        model.load_state_dict(a['state_dict'])
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        model.to(device)
        model.eval()

        # Load eval and target snapshot 
        TORCH_FLOAT = torch.float32
        if local: 
            #nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
            nrs_snap_dir = './temp'
        else:
            nrs_snap_dir = '/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/one_shot'
        
        # Load in edge index 
        poly_lo = 1
        poly_hi = 7
        Re = '1600'
        if local:
            case_path = "./temp"
        else: 
            case_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_{Re}_poly_7"
        edge_index_path_lo = f"{case_path}/gnn_outputs_poly_{poly_lo}/edge_index_element_local_rank_0_size_4"
        edge_index_path_hi = f"{case_path}/gnn_outputs_poly_{poly_hi}/edge_index_element_local_rank_0_size_4"

        if mode == "single_scale":
            edge_index_lo = get_edge_index(edge_index_path_lo)
            edge_index_hi = get_edge_index(edge_index_path_hi)
        elif mode == "multi_scale":
            edge_index_vertex_path_lo = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly_lo}/edge_index_element_local_vertex_rank_0_size_4"
            edge_index_vertex_path_hi = f"{case_path}/Re_{Re}_poly_7/gnn_outputs_poly_{poly_hi}/edge_index_element_local_vertex_rank_0_size_4"
            edge_index_lo = get_edge_index(edge_index_path_lo,
                                        edge_index_vertex_path_lo)
            edge_index_hi = get_edge_index(edge_index_path_hi,
                                        edge_index_vertex_path_hi)

        t_str_list = ['00017','00019', '00020','00021'] # 1 takes ~5 min 
        t_str_list = ['00017', '00020']
        #t_str_list = ['000%02d' %(i) for i in range(12,41)]


        # Get full edge index 
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

        for t_str in t_str_list:
            # One-shot
            xlo_field = readnek(nrs_snap_dir + f'/snapshots_coarse_{poly_hi}to{poly_lo}/newtgv0.f{t_str}')
            #xhi_field = readnek(nrs_snap_dir + f'/snapshots_interp_{poly_lo}to{poly_hi}/newtgv0.f{t_str}')
            xhi_field = xlo_field
            n_snaps = len(xlo_field.elem)


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

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"Evaluating snap {i}/{n_snaps}")
                    
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
                    data = Data( x = vel_xlo_i.to(dtype=TORCH_FLOAT),
                                      x_mean_lo = x_mean_element_lo.to(dtype=TORCH_FLOAT),
                                      x_std_lo = x_std_element_lo.to(dtype=TORCH_FLOAT),
                                      x_mean_hi = x_mean_element_hi.to(dtype=TORCH_FLOAT),
                                      x_std_hi = x_std_element_hi.to(dtype=TORCH_FLOAT),
                                      pos_norm_lo = (pos_xlo_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                                      pos_norm_hi = (pos_xhi_i/lengthscale_element).to(dtype=TORCH_FLOAT),
                                      edge_index_lo = edge_index,
                                      edge_index_hi = edge_index_hi,
                                      central_element_mask = central_element_mask,
                                      eid = torch.tensor(i))

                    # for synchronizing across element boundaries  
                    if n_element_neighbors > 0:
                        batch = None
                        edge_index_coin = ngs.get_edge_index_coincident(
                                batch, data.pos_norm_lo, data.edge_index_lo)
                        degree = utils.degree(edge_index_coin[1,:], num_nodes = data.pos_norm_lo.shape[0])
                        degree += 1.
                        data.edge_index_coin = edge_index_coin
                        data.degree = degree
                    else:
                        data.edge_index_coin = None
                        data.degree = None

                    data = data.to(device)
                    
                    # ~~~~ Model evaluation ~~~~ # 
                    
                    # ~~~~ SB: put model evaluation here!!! 

                    # ~~~~ Making the .f file ~~~~ # 
                    # Re-shape the prediction, convert back to fp64 numpy 
                    y_pred = y_pred.cpu()
                    orig_shape = x_field.elem[i].vel.shape
                    y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()

                    # Place back in the snapshot data 
                    x_field.elem[i].vel[:,:,:,:] = y_pred_rs

                # Write 
                print('Writing...')
                # incremental: 
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_full_{poly-2}to{poly}"
                #directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_{poly-2}to{poly}"
                # One shot: 
                directory_path = nrs_snap_dir + f"/snapshots_gnn_correction_{mode}_1to{poly}"
                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)
                    print(f"Directory '{directory_path}' created.")
                writenek(directory_path +  f"/newtgv0.f{t_str}", x_field)
                print(f'finished writing {t_str}') 

    # ~~~~ Save predicted flowfield into .f file 
    # Just the KNN interpolation!!! 
    if 1 == 0:
        # Load eval and target snapshot 
        TORCH_FLOAT = torch.float32
        
        xlo_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_knninterp_3to5"
        xhi_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_target"
        xknn_path = "/lus/eagle/projects/datascience/sbarwey/codes/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7_testset/incr/snapshots_knninterp_5to7"

        for snap in ["16", "17", "18", "19", "20", "21"]:
            xlo_field = readnek(f"{xlo_path}/newtgv0.f000{snap}")
            xhi_field = readnek(f"{xhi_path}/tgv0.f000{snap}")
            n_snaps = len(xlo_field.elem)

            with torch.no_grad():
                for i in range(n_snaps):
                    print(f"knn interp -- Evaluating snap {i}/{n_snaps}")
                    
                    pos_xlo_i = torch.tensor(xlo_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    vel_xlo_i = torch.tensor(xlo_field.elem[i].vel).reshape((3, -1)).T
                    pos_xhi_i = torch.tensor(xhi_field.elem[i].pos).reshape((3, -1)).T # pygeom pos format -- [N, 3] 
                    
                    # ~~~~ kNN evaluation ~~~~ # 
                    y_pred = tgnn.unpool.knn_interpolate(
                            x = vel_xlo_i,
                            pos_x = pos_xlo_i,
                            pos_y = pos_xhi_i,
                            k = 8)

                    # ~~~~ Making the .f file ~~~~ # 
                    # Re-shape the prediction, convert back to fp64 numpy 
                    y_pred = y_pred.cpu()
                    orig_shape = xhi_field.elem[i].vel.shape
                    y_pred_rs = torch.reshape(y_pred.T, orig_shape).to(dtype=torch.float64).numpy()

                    # Place back in the snapshot data 
                    xhi_field.elem[i].vel[:,:,:,:] = y_pred_rs

                # Write 
                print('Writing...')
                writenek(f"{xknn_path}/newtgv0.f000{snap}", xhi_field)

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
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
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

            sample = dataset[0]


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
  
                # loss before gnn 
                mse_before_gnn[i] = F.mse_loss(data.x, data.y)

                # loss after gnn 
                mse_after_gnn[i] = F.mse_loss(y_pred, data.y) 

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
        ax.set_yscale('log')
        plt.show(block=False)
    
    # ~~~~ Visualize model predictions -- a single element  
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
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'


        model.to(device)
        model.eval()

        with torch.no_grad():
            #data = test_dataset[4000]
            data = train_dataset[4000]
             
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

    # Comparing two models, on the full dataset testing 
    if 1 == 0:
        mode = 'multi_scale'
        data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)

        # Load data 
        train_dataset = torch.load(data_dir + "/train_dataset.pt")
        test_dataset = torch.load(data_dir + "/valid_dataset.pt")

        # Load model 1 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_3_7_128_3_2_6.tar' %(mode))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']
        model_1 = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model_1.load_state_dict(a['state_dict'])
        device = 'cpu'
        model_1.to(device)
        model_1.eval()

        # Load model 2 
        a = torch.load('./saved_models/%s/gnn_lr_1em4_unscaledResidual_3_7_128_3_2_6.tar' %(mode))
        input_dict = a['input_dict'] 
        input_node_channels = input_dict['input_node_channels']
        input_edge_channels = input_dict['input_edge_channels'] 
        hidden_channels = input_dict['hidden_channels']
        output_node_channels = input_dict['output_node_channels']
        n_mlp_hidden_layers = input_dict['n_mlp_hidden_layers']
        n_messagePassing_layers = input_dict['n_messagePassing_layers']
        name = input_dict['name']
        model_2 = gnn.GNN(input_node_channels,
                           input_edge_channels,
                           hidden_channels,
                           output_node_channels,
                           n_mlp_hidden_layers,
                           n_messagePassing_layers,
                           name)
        model_2.load_state_dict(a['state_dict'])
        device = 'cpu'
        model_2.to(device)
        model_2.eval()

        N_test = len(test_dataset)

        error_1 = torch.zeros((N_test, 3))
        error_2 = torch.zeros((N_test, 3))
        y_std = torch.zeros((N_test))

        with torch.no_grad():
            for i in range(N_test):
                print(f"iter {i} / {N_test}")
                data = test_dataset[i]
                 
                # 1) Preprocessing: scale input  
                eps = 1e-10
                x_scaled = (data.x - data.x_mean)/(data.x_std + eps)

                # 2) evaluate model 
                out_gnn_1 = model_1(x_scaled, data.edge_index, data.pos_norm, data.batch)
                out_gnn_2 = model_2(x_scaled, data.edge_index, data.pos_norm, data.batch)
                    
                # 3) get prediction: out_gnn = (data.y - data.x)/(data.x_std + eps)
                y_pred_1 = out_gnn_1 * (data.x_std + eps) + data.x 
                y_pred_2 = out_gnn_2 + data.x 

                error_1[i] = torch.mean( (y_pred_1 - data.y)**2, dim=0 ) 
                error_2[i] = torch.mean( (y_pred_2 - data.y)**2, dim=0 ) 
                y_std[i] = torch.std(data.y)


        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(1,3, figsize=(10,4))
        for comp in range(3):
            ax[comp].scatter(error_1[:,comp], y_std, color='black')
            ax[comp].scatter(error_2[:,comp], y_std, color='blue')
            ax[comp].set_title('y_std vs error, comp=%d' %(comp))
            ax[comp].set_xlabel('Error')
            ax[comp].set_ylabel('y_std')
            ax[comp].set_xscale('log')
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
        #ax.plot(values_1600_2[:,0], values_1600_2[:,2], label='Re=1600, 72^3, P=7', lw=lw)
        # ax.plot(values_2000[:,0], values_2000[:,2], label='Re=2000', lw=lw)
        # ax.plot(values_2400[:,0], values_2400[:,2], label='Re=2400', lw=lw)
        ax.set_title('Kinetic Energy')
        #ax.set_xlim([7.5, 10.5])
        plt.show(block=False)

        # dissipation rate ( dEk/dt )
        fig, ax = plt.subplots()
        ax.plot(values_1600[:,0], -values_1600[:,4], label='Re=1600, 36^3, P=7', lw=lw)
        ax.vlines(x=8,  ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=9,  ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=10, ymin =0, ymax = 0.013, color='black', lw=2)
        ax.vlines(x=8.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        ax.vlines(x=9.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        ax.vlines(x=10.5, ymin =0, ymax = 0.013, color='red', lw=2, ls='--')
        #ax.plot(values_1600_2[:,0], -values_1600_2[:,4], label='Re=1600, 72^3, P=7', lw=lw, ls='--')
        #ax.plot(values_2000[:,0], -values_2000[:,5], label='Re=2000', lw=lw)
        #ax.plot(values_2400[:,0], -values_2400[:,5], label='Re=2400', lw=lw)
        ax.set_title('Dissipation Rate')
        ax.set_xlim([7.5, 11])
        #ax.legend()
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

    # ~~~~ Compute POD basis 
    if 1 == 0:
        print('pod basis')
        # mode = "multi_scale"
        # data_dir = "/Volumes/Novus_SB_14TB/ml/DDP_PyGeom_SR/datasets/%s/Single_Snapshot_Re_1600_T_10.0_Interp_1to7/" %(mode)
        # train_dataset = torch.load(data_dir + "/train_dataset.pt")
        # test_dataset = torch.load(data_dir + "/valid_dataset.pt")

        raw_data_path = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7/snapshots_target/gnn_outputs_poly_7'
        time_str = '10.0'
        rank = 0
        SIZE = 4 
        x = torch.tensor(np.loadtxt(raw_data_path + '/fld_u_time_%s_rank_%d_size_%d' %(time_str,rank,SIZE)))
        node_element_ids = torch.tensor(np.loadtxt(raw_data_path + '/node_element_ids_rank_%d_size_%d' %(rank, SIZE), dtype=np.int64))
        x = torch.stack(utils.unbatch(x, node_element_ids, dim=0)).numpy()


        Ne = x.shape[0]
        Np = x.shape[1]
        Nf = x.shape[2]

        x = x.reshape((Ne, Np*Nf))
        
        print('Covariance...')
        cov = (1.0 / (Ne - 1.0)) * x.T @ x
        lam, evec = np.linalg.eig(cov)

        # a = np.diag(evec.T @ evec)

        fig, ax = plt.subplots()
        ax.plot(lam)
        ax.set_yscale('log')
        plt.show(block=False)
        




        # Project a single element in the POD basis 
        # ELE --- [Nf, 1]
        # MODE --- [Nf, 4]
        # A -- ELE.T @ MODE --- [1, 4]  
        # RECON -- A @ MODE.T 

        N_modes = Nf*Np 
        sample = x[0, :].reshape((1,-1)) 
        modes = evec[:, 0:N_modes]
        coeffs = (sample @ modes).squeeze()
        
        fig, ax = plt.subplots()
        ax.plot(coeffs)
        ax.set_xscale('log')
        plt.show(block=False)
 

    # ~~~~ PyMech tests -- SB: this is how we go between pymech / pygeom representations. 
    if 1 == 0:
        nrs_snap_dir = '/Volumes/Novus_SB_14TB/nek/nekrs_cases/examples_v23_gnn/tgv/Re_1600_poly_7'
        field1 = readnek(nrs_snap_dir + '/snapshots_target/newtgv0.f00010')
        first_element = field1.elem[0]
        print("Type =", type(first_element))
        print(first_element)

        #field2 = readnek(nrs_snap_dir + '/snapshots_interp_1to7/newtgv0.f00010')
        field2 = readnek(nrs_snap_dir + '/snapshots_coarse_7to1/newtgv0.f00010')

        # i_err = []
        # for i in range(len(field1.elem)):
        #     pos_1 = field1.elem[i].pos
        #     pos_2 = field2.elem[i].pos

        #     x_gll = pos_1[0,0,0,:]
        #     dx_min = x_gll[1] - x_gll[0] 

        #     # x = pos_1[0,0,0,:]
        #     # y = np.ones_like(x)
        #     # fig, ax = plt.subplots()
        #     # ax.plot(x, y, marker='o', ms=20)
        #     # plt.show(block=False)
        #     # asdf

        #     error_max = (pos_1 - pos_2).max()
        #     rel_error = (error_max / dx_min)*100

        #     if rel_error> 1e-2:
        #         print(f"i={i} \t error_max = {error_max} \t rel_error = {rel_error}")
        #         print("WARNING --- relative error in positions exceeds 0.01%")
        #         i_err.append(i)

        
        field_fine = field1
        field_crse = field2

        eid = 145 
        pos_fine = (field_fine.elem[eid].pos).reshape((3,-1)).T
        pos_crse = (field_crse.elem[eid].pos).reshape((3,-1)).T

        fig, ax = plt.subplots()
        ax.scatter(pos_fine[:,0], pos_fine[:,1])
        ax.scatter(pos_crse[:,0], pos_crse[:,1])
        plt.show(block=False)


        # construct interpolation matrices 
        #pos_fine = (field_fine.elem[eid].pos).reshape((3,-1)).T
        #pos_crse = (field_crse.elem[eid].pos).reshape((3,-1)).T
        eid = 0
        pos_fine = field_fine.elem[eid].pos 
        pos_crse = field_crse.elem[eid].pos


        # Test reshaping : 
        if 1 == 0:
            pos_1 = torch.tensor(first_element.pos).reshape((3, -1)).T # pygeom pos format -- [N, 3]
            x_1 = torch.tensor(first_element.vel).reshape((3, -1)).T # pygeom x format -- [N, 3]

            pos_orig = first_element.pos
            x_orig = first_element.vel

            pos_2 = torch.reshape(pos_1.T, (3,8,8,8))
            x_2 = torch.reshape(x_1.T, (3,8,8,8))


        # Test read/write: 
        if 1 == 0:
            # Write new file 
            x_1 = torch.tensor(first_element.vel).reshape((3, -1)).T # pygeom x format -- [N, 3]
            y = x_1 + 84.0
            y = torch.reshape(y.T, (3,8,8,8))


            # modify one lemenet data and re-write into new .f file 
            print(f"data before: {field.elem[0].vel}")
            field.elem[0].vel = y.numpy()
            print(f"data after: {field.elem[0].vel}")

            
            print("Writing...")
            writenek("./temp/tgv_override0.f00001", field)



            print("Reading again...")
            field2 = readnek("./temp/tgv_override0.f00001")
            y_test = field.elem[0].vel


        # Test plotting: 
        if 1 == 0:
            # Read in element-local edge index 
            edge_index = torch.tensor(np.loadtxt("./temp/edge_index_element_local_rank_0_size_4").astype(np.int64).T)


            # Plot 
            pos = torch.tensor(first_element.pos).reshape((3, -1)).T
            edge_xyz = pos[edge_index].permute(1,0,2)
            
            ms = 50
            lw_edge = 2
            lw_marker = 1
            fig = plt.figure(figsize=(12,8))
            ax = fig.add_subplot(111, projection="3d")

            # Plot the edges
            count = 0
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="black", lw=lw_edge, alpha=0.3)
                #ax.plot(*vizedge.T, color="black", lw=lw_edge * edge_weights[RANK][count], alpha=0.3)
                count += 1
            ax.scatter(*pos.T, s=ms, ec='black', lw=lw_marker, c='black', alpha=1)
            
            ax.grid(False)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=34, azim=-24, roll=0)
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.show(block=False)
        

