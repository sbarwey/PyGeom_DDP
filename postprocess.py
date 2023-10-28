import numpy as np
import os,sys,time
import torch 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data 
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn 
import matplotlib.pyplot as plt

Tensor = torch.Tensor 
TORCH_FLOAT = torch.float32
NP_FLOAT = np.float32
TORCH_INT = torch.int64
NP_INT = np.int64

DIM = 3 
SIZE = 4
#CASE_PATH = '/Users/sbarwey/Files/nekrs_cases/examples_v23/tgv_gnn'
CASE_PATH = '/Users/sbarwey/Files/nekrs_cases/examples_v23/tgv'
P_HI = 7 
P_LO = 7 

def get_pygeom_dataset(data_path: str,
                       rank: int) -> Data:
    print('Loading data: %s, rank %s' %(data_path, rank))
    t_load = time.time()
    edge_index = np.loadtxt(data_path + '/edge_index_element_local_rank_%d_size_%d' %(rank, SIZE), 
                            dtype=NP_INT).T 
    node_element_ids = np.loadtxt(data_path + '/node_element_ids_rank_%d_size_%d' %(rank, SIZE), dtype=NP_INT)
    global_ids = np.loadtxt(data_path + '/global_ids_rank_%d_size_%d' %(rank, SIZE), dtype=NP_INT)
    pos = np.loadtxt(data_path + '/pos_node_rank_%d_size_%d' %(rank, SIZE), dtype=NP_FLOAT)
    x = np.loadtxt(data_path + '/fld_u_rank_%d_size_%d' %(rank,SIZE), dtype=NP_FLOAT)

    # Make tensor 
    edge_index = torch.tensor(edge_index)
    node_element_ids = torch.tensor(node_element_ids)
    global_ids = torch.tensor(global_ids)
    pos = torch.tensor(pos)
    x = torch.tensor(x)

    # # un-batch elements 
    # utils.unbatch(pos, node_element_ids, dim=0)
    # utils.unbatch(x, node_element_ids, dim=0)

    # Make data 
    data = Data(x=x, 
                pos=pos, 
                edge_index=edge_index, 
                node_element_ids=node_element_ids, 
                global_ids=global_ids,
                batch=node_element_ids)
    t_load = time.time() - t_load 
    print('\ttook %g sec' %(t_load))
    return data 


def get_rms(x_batch: Tensor) -> Tensor:
    u_var = x_batch.var(dim=1, keepdim=True)
    tke = 0.5*u_var.sum(dim=2)
    u_rms = torch.sqrt(tke / 1.5)
    return u_rms 


def ratio_boundary_internal_nodes(p: int) -> float: 
    ratio = ( (p+1)**3 - (p-1)**3 )/(p-1)**3
    return ratio 

if __name__ == "__main__":

    rank = 0
    data_path_hi = CASE_PATH + '/gnn_outputs_original_poly_%d' %(P_HI)
    data_hi = get_pygeom_dataset(data_path_hi, rank)
    data_path_lo = CASE_PATH + '/gnn_outputs_recon_poly_%d' %(P_LO)
    data_lo = get_pygeom_dataset(data_path_lo, rank)
   
    # Split into multiple graphs based on element ids  
    pos_hi = torch.stack(utils.unbatch(data_hi.pos, data_hi.node_element_ids, dim=0))
    x_hi = torch.stack(utils.unbatch(data_hi.x, data_hi.node_element_ids, dim=0))
    gli_hi = torch.stack(utils.unbatch(data_hi.global_ids, data_hi.node_element_ids, dim=0))

    pos_lo = torch.stack(utils.unbatch(data_lo.pos, data_lo.node_element_ids, dim=0))
    x_lo = torch.stack(utils.unbatch(data_lo.x, data_lo.node_element_ids, dim=0))
    gli_lo = torch.stack(utils.unbatch(data_lo.global_ids, data_lo.node_element_ids, dim=0))


    # ~~~~ Postprocessing 
    # get rms 
    rms_hi = get_rms(x_hi)
    rms_lo = get_rms(x_lo)

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

    # ~~~~ Plotting 
    if 1 == 1:
        print('Plotting...')

        # Plot the edges 
        if 1 == 0:
            
            #for rl in range(SIZE):
            data = Data(pos = pos_hi[0], x = pos_hi[0], edge_index = data_hi.edge_index) 
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
        



