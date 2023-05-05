"""
PyTorch DDP integrated with PyGeom for multi-node training
"""
from __future__ import absolute_import, division, print_function, annotations
import os
import socket
import logging

from typing import Optional, Union, Callable

import numpy as np

import hydra
import time
import torch
import torch.utils.data
import torch.utils.data.distributed
from torch.cuda.amp.grad_scaler import GradScaler
import torch.multiprocessing as mp
import torch.distributions as tdist 

import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
Tensor = torch.Tensor

# PyTorch Geometric
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn

# Plotting
import networkx as nx
import matplotlib.pyplot as plt

# Models
import models.cnn as cnn
import models.gnn as gnn

# Data preparation
import dataprep.unstructured_mnist as umnist
import dataprep.backward_facing_step as bfs



log = logging.getLogger(__name__)

# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0')
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'cpu'
    if DEVICE == 'gpu':
        DEVICE_ID = 'cuda:0' 
    else:
        DEVICE_ID = 'cpu'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)



def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup():
    dist.destroy_process_group()


def metric_average(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val


class Trainer:
    def __init__(self, cfg: DictConfig, scaler: Optional[GradScaler] = None):
        self.cfg = cfg
        self.rank = RANK
        if scaler is None:
            self.scaler = None
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.backend = self.cfg.backend
        if WITH_DDP:
            init_process_group(RANK, SIZE, backend=self.backend)
        
        # ~~~~ Init torch stuff 
        self.setup_torch()

        # ~~~~ Setup data 
        self.neighboring_procs = {}
        self.data, self.data_gll = self.setup_data()

        self.n_nodes_internal_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        if WITH_CUDA:
            self.data.n_nodes_internal = self.data.n_nodes_internal.cuda()
        dist.all_gather(self.n_nodes_internal_procs, self.data.n_nodes_internal)
        print('[RANK %d] -- data: ' %(RANK), self.data)

        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()

        # ~~~~ Initialize send/recv buffers on device (if applicable)
        n_features_pos = self.data.pos.shape[1]
        self.pos_buffer_send, self.pos_buffer_recv = self.build_buffers(n_features_pos)

        # ~~~~ Do a halo swap on position matrices 
        #if RANK == 1: 
        #    print('[RANK %d] -- pos before: ' %(RANK), self.data.pos)
        if WITH_CUDA:
            self.data.pos = self.data.pos.cuda()
        self.data.pos = self.halo_swap(self.data.pos, self.pos_buffer_send, self.pos_buffer_recv)
        #if RANK == 0: 
        #    print('[RANK %d] -- pos after: ' %(RANK), self.data.pos)

        # ~~~~ Init model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # # ~~~~ Wrap model in DDP
        # if WITH_DDP and SIZE > 1:
        #     self.model = DDP(self.model)

    def build_model(self) -> nn.Module:
        input_channels = self.data.x.shape[1]
        hidden_channels = input_channels
        model = gnn.Simple_MP_Layer(input_channels=input_channels, 
                                    hidden_channels=hidden_channels)
        return model

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def halo_swap(self, input_tensor, buff_send, buff_recv):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:
            # Fill send buffer
            for i in self.neighboring_procs:
                buff_send[i] = input_tensor[self.mask_send[i]]

            # Perform swap
            req_send_list = []
            for i in self.neighboring_procs:
                req_send = dist.isend(tensor=buff_send[i], dst=i)
                req_send_list.append(req_send)
            
            req_recv_list = []
            for i in self.neighboring_procs:
                req_recv = dist.irecv(tensor=buff_recv[i], src=i)
                req_recv_list.append(req_recv)

            for req_send in req_send_list:
                req_send.wait()

            for req_recv in req_recv_list:
                req_recv.wait()

            dist.barrier()

            # Fill halo nodes 
            for i in self.neighboring_procs:
                input_tensor[self.mask_recv[i]] = buff_recv[i]
        return input_tensor 

    def build_masks(self):
        """
        Builds index masks for facilitating halo swap of nodes 
        """
        mask_send = [None] * SIZE
        mask_recv = [None] * SIZE
        if SIZE > 1: 
            n_nodes_local = self.data.n_nodes_internal + self.data.n_nodes_halo
            halo_info = self.data.halo_info

            for i in self.neighboring_procs:
                idx_i = self.data.halo_info[:,2] == i
                # index of nodes to send to proc i 
                mask_send[i] = self.data.halo_info[:,0][idx_i] 
                #mask_send[i] = torch.unique(mask_send[i])
                
                # index of nodes to receive from proc i  
                mask_recv[i] = self.data.halo_info[:,1][idx_i]
                #mask_recv[i] = torch.unique(mask_recv[i])


            print('[RANK %d] mask_send: ' %(RANK), mask_send)
            print('[RANK %d] mask_recv: ' %(RANK), mask_recv)

        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        buff_send = [None] * SIZE
        buff_recv = [None] * SIZE
        if SIZE > 1: 
            for i in self.neighboring_procs:
                buff_send[i] = torch.empty([len(self.mask_send[i]), n_features], dtype=torch.float32, device=DEVICE_ID) 
                buff_recv[i] = torch.empty([len(self.mask_recv[i]), n_features], dtype=torch.float32, device=DEVICE_ID)
        return buff_send, buff_recv 

    def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
        """
        Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
        NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
        n_internal_nodes can vary for each proc, but n_features must be the same 
        """
        # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
        n_nodes = torch.tensor(input_tensor.shape[0])
        n_features = torch.tensor(input_tensor.shape[1])

        n_nodes_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        if WITH_CUDA:
            n_nodes = n_nodes.cuda()
        dist.all_gather(n_nodes_procs, n_nodes)



        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty([n_nodes_procs[i], n_features], 
                                             dtype=dtype,
                                             device=DEVICE_ID)
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list

    # def gather_node_tensor(self, input_tensor, dst=0, dtype=torch.float32):
    #     """
    #     Gathers node-based tensor into root proc. Shape is [n_internal_nodes, n_features] 
    #     NOTE: input tensor on all ranks should correspond to INTERNAL nodes (exclude halo nodes) 
    #     n_internal_nodes can vary for each proc, but n_features must be the same 
    #     """
    #     # torch.distributed.gather(tensor, gather_list=None, dst=0, group=None, async_op=False)
    #     n_features = input_tensor.shape[1]
    #     gather_list = None
    #     if RANK == 0:
    #         gather_list = [None] * SIZE
    #         for i in range(SIZE):
    #             gather_list[i] = torch.empty([self.n_nodes_internal_procs[i], n_features],
    #                                          dtype=dtype,
    #                                          device=DEVICE_ID)
    #     dist.gather(input_tensor, gather_list, dst=0)
    #     return gather_list 
        
    
    def setup_data(self):
        """
        Creates 1d graph which is partitioned to each proc  
        """
        # ~~~~ Create simple 1d graph
        # paths:
        main_path = self.cfg.data_dir + 'wall/data_halo/elements_32_poly_5_nproc_%d/' %(SIZE)
        path_to_pos_elem = main_path + 'pos_element_proc_%d' %(RANK)
        path_to_pos_gll = main_path + 'pos_node_proc_%d' %(RANK)
        path_to_ei_elem = main_path + 'ei_element_proc_%d' %(RANK)
        path_to_ei_gll2elem = main_path + 'ei_node2element_proc_%d' %(RANK)
        path_to_sol_gll = main_path + 'u_node_proc_%d' %(RANK)
        path_to_halo_info = main_path + 'halo_info_proc_%d' %(RANK)

        # Load
        pos_elem = np.loadtxt(path_to_pos_elem, dtype=np.float32) # element positions
        pos_gll = np.loadtxt(path_to_pos_gll, dtype=np.float32) # node positions 
        ei_elem = np.loadtxt(path_to_ei_elem, dtype=np.int64).T 
        ei_gll2elem = np.loadtxt(path_to_ei_gll2elem, dtype=np.int64).T
        sol_gll = np.loadtxt(path_to_sol_gll, dtype=np.float32)
        halo_info = np.loadtxt(path_to_halo_info, dtype=np.int64) 
        print('[RANK %d] Halo info: ' %(RANK), halo_info.shape)

        if SIZE > 1: 
            # Get list of neighboring processors for each processor
            self.neighboring_procs = np.unique(halo_info[:,2])
            print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)

            # Get number of internal and halo nodes 
            n_nodes_internal = pos_elem.shape[0]
            n_nodes_halo = halo_info.shape[0]
            n_nodes = n_nodes_internal + n_nodes_halo
            idx_internal_nodes = list(range(n_nodes_internal))
            idx_halo_nodes = list(range(n_nodes_internal, n_nodes))
            #print('[RANK %d] idx_internal_nodess: ' %(RANK), idx_internal_nodes)
        else:
            print('[RANK %d] neighboring procs: ' %(RANK), self.neighboring_procs)

            # Get number of internal and halo nodes 
            n_nodes_internal = pos_elem.shape[0]
            n_nodes_halo = 0
            n_nodes = n_nodes_internal + n_nodes_halo
            idx_internal_nodes = list(range(n_nodes_internal))
            idx_halo_nodes = [] 
            #print('[RANK %d] idx_internal_nodess: ' %(RANK), idx_internal_nodes)

        # Get local node attribute matrix 
        n_features = 3 
        x = torch.zeros((n_nodes, n_features))
        x[idx_internal_nodes] = torch.ones(n_nodes_internal, n_features)
        pos = torch.zeros((n_nodes, pos_elem.shape[1]))
        pos[idx_internal_nodes] = torch.tensor(pos_elem[:])

        # set x = pos 
        x[idx_internal_nodes] = pos[idx_internal_nodes]

        # Make local graph -- element based 
        data_local = Data(x=x, edge_index=torch.tensor(ei_elem), pos=pos, 
                          n_nodes_internal=torch.tensor(n_nodes_internal, dtype=torch.int64),
                          n_nodes_halo=torch.tensor(n_nodes_halo, dtype=torch.int64), 
                          idx_internal_nodes = torch.tensor(idx_internal_nodes, dtype=torch.int64),
                          idx_halo_nodes = torch.tensor(idx_halo_nodes, dtype=torch.int64),
                          halo_info = torch.tensor(halo_info, dtype=torch.int64)) 
        data_local = data_local.to(DEVICE_ID)


        # Make local graph -- node based 
        data_local_gll = Data(x = torch.tensor(sol_gll), 
                              pos = torch.tensor(pos_gll), 
                              cluster = torch.tensor(ei_gll2elem[1,:]))
        data_local_gll = data_local_gll.to(DEVICE_ID)

        # Create a nearest-neighbors graph, k = 3 
        data_local_gll.edge_index = tgnn.knn_graph(data_local_gll.pos, k = 3)
                              
        print('cluster:', data_local_gll.cluster)

        # Coalesce:
        data_local.edge_index = utils.coalesce(data_local.edge_index)
        data_local.edge_index = utils.to_undirected(data_local.edge_index)

        return data_local, data_local_gll


    def plot_graph(self, plot_3d=True):
        data_element = Data(x = self.data.pos, edge_index = self.data.edge_index, pos = self.data.pos)
        G_element = utils.to_networkx(data=data_element)

        fig = plt.figure()
        if plot_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111) 

        data = data_element
        G = G_element
        pos = dict(enumerate(np.array(data.pos)))

        # Extract node and edge positions from the layout
        node_xyz = np.array([pos[v] for v in sorted(G)])
        edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

        # idx_internal: 
        idx_internal = self.data.idx_internal_nodes
        idx_halo = self.data.idx_halo_nodes

        # Plot the nodes - alpha is scaled by "depth" automatically
        if plot_3d:
            ax.scatter(*node_xyz[idx_internal].T, s=100, ec="w", c='black')
            ax.scatter(*node_xyz[idx_halo].T, s=100, ec="w", c='red')
        
            # Plot the edges
            for vizedge in edge_xyz:
                ax.plot(*vizedge.T, color="tab:gray")

            # # Overlay the GLL points 
            # gll = data_node.pos
            # ax.scatter(*gll.T, s=5, color="red", alpha=0.2)

        else:
            ax.scatter(*node_xyz[idx_internal,:2].T, s=50, c='black')
            ax.scatter(*node_xyz[idx_halo,:2].T, s=50, c='red')
        
            # Plot the edges
            for vizedge in edge_xyz[:,:,:2]:
                ax.plot(*vizedge.T, color="lime")

            ## Overlay the GLL points 
            #gll = data_node.pos[:,:2]
            #ax.scatter(*gll.T, s=15, color="red")

            ## Plot arrows from GLL points to element
            #cluster = cluster_node_list[i][1,:]
            #n_points = data_node.pos.shape[0]
            #for p in range(n_points):
            #    sample = gll[p]
            #    centroid = data.pos[cluster[p]]
            #    ax.arrow(sample[0], sample[1], centroid[0] - sample[0], centroid[1] - sample[1],
            #             length_includes_head=True, width=0.01, head_width=0.1, color='black', zorder=0)
        

        lo = -np.pi - 0.1
        hi = np.pi + 0.1
        def _format_axes(ax):
            """Visualization options for the 3D axes."""
            # Turn gridlines off
            ax.grid(False)
            # Suppress tick labels
            #for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            #    dim.set_ticks([])
            # Set axes labels
            ax.set_xlabel("x")
            ax.set_xlim([lo, hi])
            ax.set_ylabel("y")
            ax.set_ylim([lo, hi])
            if plot_3d:
                ax.set_zlabel("z")
                ax.set_zlim([lo, hi])
            else:
                ax.set_aspect('equal')

        _format_axes(ax)
        fig.tight_layout()
        save_dir = self.cfg.work_dir + '/outputs'
        plt.savefig(save_dir + '/graph_proc_%d.png' %(RANK), dpi=800)
        plt.close()

        return 0


    def plot_field(self, field, name='test'):

        pos = self.data_gll.pos
        fig, ax = plt.subplots()
        ax.scatter(pos[:,0], pos[:,1], c=field, s=5) 
        ax.set_aspect('equal')
        save_dir = self.cfg.work_dir + '/outputs'
        ax.grid(False)
        plt.savefig(save_dir + '/%s_node_field_%d.png' %(name,RANK), dpi=800)
        plt.close()

        return


    def run_mp(self, n_mp):
        self.model.eval()
        with torch.no_grad(): 
            if WITH_CUDA: 
                self.data.x = self.data.x.cuda()
                self.data.edge_index = self.data.edge_index.cuda()
                self.data.pos = self.data.pos.cuda()

            # gll-to-element encoder: 
            x = self.data_gll.x
            ei_gll = self.data_gll.edge_index
            cluster = self.data_gll.cluster
            x = self.model.run_encoder(x = x, 
                                       edge_index = ei_gll, 
                                       cluster = cluster)
            x = torch.concat((x, 
                              torch.zeros(self.data.n_nodes_halo, self.model.hidden_channels)), axis=0)

            #x = self.data.x
            
            # build buffers  
            x_buffer_send, x_buffer_recv = self.build_buffers(self.model.hidden_channels)

            # message passing loop:
            for i in range(n_mp):
                #x = self.halo_swap(x, x_buffer_send, 
                #                   x_buffer_recv)
                x = self.model.run_messagepassing(x, 
                                        self.data.edge_index)

            ## element-to-gll decoder: 
            #x = self.halo_swap(x, 
            #                   x_buffer_send, 
            #                   x_buffer_recv)
            #x = self.model.run_decoder(x, 
            #                           self.data.pos, 
            #                           self.data_gll.pos)

        
        return x  

def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)

def message_passing(cfg: DictConfig):
    trainer = Trainer(cfg)

    # trainer.plot_graph(plot_3d=False)

    # trainer.plot_field(trainer.data_gll.x[:,0], name='input')
    
    n_mp = 10 # number of message passing steps 
    out = trainer.run_mp(n_mp) 

    # trainer.plot_field(out[:,0], name='output')

    # gather node tensor 
    input_tensor = out # trainer.data.pos
    input_tensor = input_tensor[trainer.data.idx_internal_nodes, :]
    print('[RANK %d] input_tensor: ' %(RANK), input_tensor.shape)
    out_full = trainer.gather_node_tensor(input_tensor)

    if RANK == 0:
        out_full = torch.cat(out_full, dim=0)
        out_full = out_full.cpu()
        print('out_full: ', out_full)

        # save: 
        out_full = out_full.numpy()
        savepath = cfg.work_dir + '/outputs/halo_results/'
        #filename = 'nprocs_%d.npy' %(SIZE)
        filename = 'nprocs_%d_nohalo.npy' %(SIZE)
        np.save(savepath + filename, out_full)


    
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    message_passing(cfg)
    cleanup()

if __name__ == '__main__':
    main()












