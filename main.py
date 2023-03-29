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
        self.data = self.setup_data()
        self.n_nodes_internal_procs = list(torch.empty([1], dtype=torch.int64, device=DEVICE_ID)) * SIZE
        if WITH_CUDA:
            self.data.n_nodes_internal = self.data.n_nodes_internal.cuda()
        dist.all_gather(self.n_nodes_internal_procs, self.data.n_nodes_internal)
        print('[RANK %d] -- data: ' %(RANK), self.data)

        # ~~~~ Setup halo exchange masks
        self.mask_send, self.mask_recv = self.build_masks()

        # ~~~~ Initialize send/recv buffers on device (if applicable)
        n_features_x = self.data.x.shape[1]
        n_features_pos = self.data.pos.shape[1]
        self.x_buffer_send, self.x_buffer_recv = self.build_buffers(n_features_x)
        self.pos_buffer_send, self.pos_buffer_recv = self.build_buffers(n_features_pos)

        # ~~~~ Do a halo swap on position matrices 
        if WITH_CUDA:
            self.data.pos = self.data.pos.cuda()
        self.data.pos = self.halo_swap(self.data.pos, self.pos_buffer_send, self.pos_buffer_recv)
        
        # ~~~~ Init model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

    def build_model(self) -> nn.Module:
        n_features = self.data.x.shape[1]
        model = gnn.Simple_MP_Layer(hidden_channels=n_features)
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
            for i in self.neighboring_procs[RANK]:
                buff_send[i] = input_tensor[self.mask_send[i]]

            # Perform swap
            req_send_list = []
            for i in self.neighboring_procs[RANK]:
                req_send = dist.isend(tensor=buff_send[i], dst=i)
                req_send_list.append(req_send)
            
            req_recv_list = []
            for i in self.neighboring_procs[RANK]:
                req_recv = dist.irecv(tensor=buff_recv[i], src=i)
                req_recv_list.append(req_recv)

            for req_send in req_send_list:
                req_send.wait()

            for req_recv in req_recv_list:
                req_recv.wait()

            dist.barrier()

            # Fill halo nodes 
            for i in self.neighboring_procs[RANK]:
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
            halo = self.data.halo

            for i in self.neighboring_procs[RANK]:
                if RANK == 0: 
                    mask_send[i] = [n_nodes_local-halo-1]
                    mask_recv[i] = [n_nodes_local-halo]
                elif RANK == SIZE-1:
                    mask_send[i] = [halo]
                    mask_recv[i] = [0]
                else:
                    if i == RANK - 1: #neighbor is on left  
                        mask_send[i] = [halo]
                        mask_recv[i] = [0]
                    elif i == RANK + 1: # neighbor is on right  
                        mask_send[i] = [n_nodes_local-halo-1]
                        mask_recv[i] = [n_nodes_local-halo]
            #print('[RANK %d] mask_send: ' %(RANK), mask_send)
            #print('[RANK %d] mask_recv: ' %(RANK), mask_recv)

        return mask_send, mask_recv 

    def build_buffers(self, n_features):
        buff_send = [None] * SIZE
        buff_recv = [None] * SIZE
        if SIZE > 1: 
            for i in self.neighboring_procs[RANK]:
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
        n_features = input_tensor.shape[1]
        gather_list = None
        if RANK == 0:
            gather_list = [None] * SIZE
            for i in range(SIZE):
                gather_list[i] = torch.empty([self.n_nodes_internal_procs[i], n_features], 
                                             dtype=dtype,
                                             device=DEVICE_ID)
        dist.gather(input_tensor, gather_list, dst=0)
        return gather_list
        
    
    def setup_data(self):
        """
        Creates 1d graph which is partitioned to each proc  
        """
        # ~~~~ Create simple 1d graph
        # number of nodes and edges
        n_nodes_global = 128
        n_edges_global = n_nodes_global - 1 

        # Node positions and attributes 
        pos = torch.linspace(0,1,n_nodes_global).reshape((n_nodes_global, 1)) 
        n_features = 4
        x = torch.rand(n_nodes_global,n_features)

        # Edge owner and neighbor 
        edge_own = torch.arange(n_nodes_global - 1)
        edge_nei = torch.arange(1, n_nodes_global)

        # Edge index 
        edge_index = torch.zeros((2, n_edges_global), dtype=torch.long)
        edge_index[0,:] = edge_own
        edge_index[1,:] = edge_nei

        # Make undirected:
        edge_index = torch_geometric.utils.to_undirected(edge_index)
        n_edges_global = edge_index.shape[1]

        idx_internal_nodes = list(range(n_nodes_global))
        data = Data(x=x, edge_index=edge_index, pos=pos, 
                    n_nodes_internal=torch.tensor(n_nodes_global, dtype=torch.int64), 
                    n_nodes_halo=torch.tensor(0, dtype=torch.int64), 
                    halo=torch.tensor(0, dtype=torch.int64),
                    idx_internal_nodes = torch.tensor(idx_internal_nodes, dtype=torch.int64)) 
        data = data.to(DEVICE_ID)

        if SIZE > 1: 
            # ~~~~ Partition the 1d graph:
            n_nodes_internal = n_nodes_global / SIZE
            assert n_nodes_internal.is_integer(), "number of nodes must be divisible by number of processors!"
            n_nodes_internal = int(n_nodes_internal)

            # starting end ending indices per processor in the node attribute matrix:
            start_index = RANK * n_nodes_internal # [int(i * nodes_local) for i in range(n_procs)]
            end_index = start_index + n_nodes_internal

            # List of neighboring processors for each processor
            for i in range(SIZE):
                if i == 0:
                    self.neighboring_procs[i] = [1] 
                elif i == SIZE - 1:
                    self.neighboring_procs[i] = [SIZE - 2]
                else:
                    self.neighboring_procs[i] = [i - 1, i + 1]

            # ~~~~ Create the local graphs 
            halo = 1 
            if RANK == 0: # Left boundary
                n_nodes_halo = halo
                n_nodes_local = n_nodes_internal + n_nodes_halo
                x_local                     = torch.zeros((n_nodes_local, n_features))
                pos_local                   = torch.zeros((n_nodes_local, n_features))
                x_local[:-halo]           = data.x[start_index:end_index]
                pos_local[:-halo]         = data.pos[start_index:end_index]
                idx_internal_nodes = list(range(n_nodes_local - halo))
            elif RANK == SIZE - 1: # right boundary
                n_nodes_halo = halo
                n_nodes_local = n_nodes_internal + n_nodes_halo
                x_local                     = torch.zeros((n_nodes_local, n_features))
                pos_local                   = torch.zeros((n_nodes_local, n_features))
                x_local[halo:]            = data.x[start_index:end_index]
                pos_local[halo:]          = data.pos[start_index:end_index]
                idx_internal_nodes = list(range(halo,n_nodes_local))
            else: # internal
                n_nodes_halo = 2*halo
                n_nodes_local = n_nodes_internal + n_nodes_halo
                x_local                     = torch.zeros((n_nodes_local, n_features))
                pos_local                   = torch.zeros((n_nodes_local, n_features))
                x_local[halo:-halo]     = data.x[start_index:end_index]
                pos_local[halo:-halo]   = data.pos[start_index:end_index]
                idx_internal_nodes = list(range(halo, n_nodes_local - halo))
            
            # Local connectivities: 
            edge_own_local = torch.arange(n_nodes_local - 1)
            edge_nei_local = torch.arange(1, n_nodes_local)

            # Edge index 
            n_edges_local = n_nodes_local - 1 
            edge_index_local = torch.zeros((2, n_edges_local), dtype=torch.long)
            edge_index_local[0,:] = edge_own_local
            edge_index_local[1,:] = edge_nei_local
            edge_index_local = torch_geometric.utils.to_undirected(edge_index_local)
            n_edges_local = edge_index_local.shape[1]

            # Make local graph
            data_local = Data(x=x_local, edge_index=edge_index_local, pos=pos_local, 
                              n_nodes_internal=torch.tensor(n_nodes_internal, dtype=torch.int64),
                              n_nodes_halo=torch.tensor(n_nodes_halo, dtype=torch.int64), 
                              halo=torch.tensor(halo, dtype=torch.int64),
                              idx_internal_nodes = torch.tensor(idx_internal_nodes, dtype=torch.int64)) 
            data_local = data_local.to(DEVICE_ID)
        else:
            data_local = data

        return data_local



    def run_mp(self, n_mp):
        self.model.eval()
        with torch.no_grad(): 
            if WITH_CUDA: 
                self.data.x = self.data.x.cuda()
                self.data.edge_index = self.data.edge_index.cuda()
                self.data.pos = self.data.pos.cuda()

            # run message passing 
            x = self.data.x
            for i in range(n_mp):
                x = self.halo_swap(x, self.x_buffer_send, self.x_buffer_recv)
                x = self.model(x, self.data.edge_index)


        return x  

def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)

def message_passing(cfg: DictConfig):
    trainer = Trainer(cfg)
    
    n_mp = 10 # number of message passing steps 
    out = trainer.run_mp(n_mp) 
    #print('[RANK %d] out: ' %(RANK), out)

    # gather node tensor 
    input_tensor = out
    halo = trainer.data.halo
    input_tensor = input_tensor[trainer.data.idx_internal_nodes, :]
    out_full = trainer.gather_node_tensor(input_tensor)
    if RANK == 0:
        out_full = torch.cat(out_full, dim=0)
        #print('out_full: ', out_full)

        # save: 
        out_full = out_full.numpy()
        savepath = cfg.work_dir + '/outputs/halo_results/'
        filename = 'nprocs_%d.npy' %(SIZE)
        np.save(savepath + filename, out_full)


    
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    message_passing(cfg)
    cleanup()

if __name__ == '__main__':
    main()












