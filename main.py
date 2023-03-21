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
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

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

        # ~~~~ Init model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

    def build_model(self) -> nn.Module:
        
        model = gnn.GCN()
        
        return model

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def setup_data(self):
        train_sampler = []
        train_loader = []
        test_sampler = [] 
        test_loader = []

        return {
            'train': {
                'sampler': train_sampler,
                'loader': train_loader,
            },
            'test': {
                'sampler': test_sampler,
                'loader': test_loader,
            }
        }

def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)

def train(cfg: DictConfig):
    trainer = Trainer(cfg)
    
@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    train(cfg)


    # # Non-blocking point-to-point communication 
    # test_tensor = torch.zeros(1)
    # if DEVICE == 'gpu':
    #     test_tensor = test_tensor.cuda()

    # req = None
    # if RANK == 0:
    #     test_tensor += 10.3

    #     # Send tensor to process 1 
    #     req = dist.isend(tensor=test_tensor, dst=1)
    #     print('Rank 0 started sending')
    # else:
    #     req = dist.irecv(tensor=test_tensor, src=0)
    #     print('Rank 1 started receiving')
    # req.wait()
    # print('Rank ', RANK, ' has data ', test_tensor[0])

    # ~~~~ Create simple 1d graph
    # number of nodes and edges
    n_nodes = 32
    n_edges = n_nodes - 1 

    # Node positions and attributes 
    pos = torch.linspace(0,1,n_nodes).reshape((n_nodes, 1)) 
    x = torch.rand(n_nodes,1)

    # Edge owner and neighbor 
    edge_own = torch.arange(n_nodes - 1)
    edge_nei = torch.arange(1, n_nodes)

    # Edge index 
    edge_index = torch.zeros((2, n_edges), dtype=torch.long)
    edge_index[0,:] = edge_own
    edge_index[1,:] = edge_nei

    # Make undirected:
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    n_edges = edge_index.shape[1]
    data = Data(x=x, edge_index=edge_index, pos=pos) 
    if DEVICE == 'gpu':
        data = data.to('cuda:0')

    # ~~~~ Partition the simple 1d graph:
    n_procs = SIZE  # number of processors

    # number of nodes per processor: 
    n_nodes_local = n_nodes / n_procs
    assert n_nodes_local.is_integer(), "number of nodes must be divisible by number of processors!"
    n_nodes_local = int(n_nodes_local)

    # starting end ending indices per processor in the node attribute matrix:
    start_index = RANK * n_nodes_local # [int(i * nodes_local) for i in range(n_procs)]
    print('start_index: ', start_index)

    # List of neighboring processors for each processor
    neighboring_procs = {}
    for i in range(n_procs):
        if i == 0:
            neighboring_procs[i] = [1] 
        elif i == n_procs - 1:
            neighboring_procs[i] = [n_procs - 2]
        else:
            neighboring_procs[i] = [i - 1, i + 1]



    # ~~~~ Create the local graphs. 
    # 1) Partition the node attribute matrix
    # 2) Partition the node position matrix 
    # 3) Create the edge index 
    print('data.x: ', data.x.shape)
    x_local = data.x[start_index:n_nodes_local]
    print('x_local, rank %d: ' %(RANK), x_local)






    # ~~~~ Create sender and receiver mask:
    # mask_send: boundary cell indices used to populate ghost cells 
    # mask_recv: ghost cell indices used to receive boundary cell info     




    #cleanup()

if __name__ == '__main__':
    main()












