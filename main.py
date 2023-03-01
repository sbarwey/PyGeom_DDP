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

# Models
import models.cnn as cnn
import models.gnn as gnn

# Data preparation
import dataprep.unstructured_mnist as umnist



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

        self.setup_torch()
        self.data = self.setup_data()
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

        self.loss_fn = nn.MSELoss()
        self.optimizer = self.build_optimizer(self.model)
        self.scheduler = self.build_scheduler(self.optimizer)
        self.epoch_start = 1

        # Restart from checkpoint
        if self.cfg.restart:
            try:
                ckpt_path = cfg.ckpt_dir + self.model.get_save_header()
            except (AttributeError) as e:
                ckpt_path = cfg.ckpt_dir + 'checkpoint.pt'
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            #self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            self.epoch_start = ckpt['epoch']
            if RANK == 0: 
                astr = 'RESTARTING FROM CHECKPOINT -- AT EPOCH %d/%d' %(self.epoch_start, self.cfg.epochs)
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)

        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()

    def build_model(self) -> nn.Module:
        model = gnn.MP_GNN(in_channels_node=1,
                               in_channels_edge=3,
                               hidden_channels=8,
                               out_channels=1,
                               n_mlp_encode=3,
                               n_mlp_mp=2,
                               n_mp=2,
                               act=F.elu)
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # DDP: scale learning rate by the number of GPUs
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-7, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)

    def setup_data(self):
        kwargs = {}
        #if self.device == 'gpu':
        #    kwargs = {'num_workers': 1, 'pin_memory': True}

        device_for_loading = 'cpu'
        train_dataset, test_dataset = umnist.get_mnist_dataset(
            self.cfg.path_to_vtk, self.cfg.path_to_ei, self.cfg.path_to_ea, 
            self.cfg.path_to_pos, device_for_loading
        )


        # DDP: use DistributedSampler to partition training data
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=SIZE, rank=RANK,
        )
        train_loader = torch_geometric.loader.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            sampler=train_sampler,
            **kwargs
        )

        # DDP: use DistributedSampler to partition the test data
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=SIZE, rank=RANK
        )
        test_loader = torch_geometric.loader.DataLoader(
            test_dataset, batch_size=self.cfg.test_batch_size
        )

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

    def train_step(
        self,
        data: DataBatch
    ) -> Tensor:
        if WITH_CUDA:
            data = data.cuda()
# 
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
        loss = self.loss_fn(out, data.x)

        if self.scaler is not None and isinstance(self.scaler, GradScaler):
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss

    def train_epoch(
            self,
            epoch: int,
    ) -> dict:
        self.model.train()
        start = time.time()
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()

        train_sampler = self.data['train']['sampler']
        train_loader = self.data['train']['loader']
        # DDP: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        for bidx, data in enumerate(train_loader):
            #print('Rank %d, bid %d, data:' %(RANK, bidx), data.batch.type())
            loss = self.train_step(data)
            running_loss += loss.item()
            count += 1
            
            # Log on Rank 0:
            if bidx % self.cfg.logfreq == 0 and RANK == 0:
                # DDP: use train_sampler to determine the number of
                # examples in this workers partition
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - start,
                    'batch_loss': loss.item(),
                    'running_loss': running_loss,
                }
                pre = [
                    f'[{RANK}]',
                    (   # looks like: [num_processed/total (% complete)]
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)}/{len(train_sampler)}'
                        f' ({100. * bidx / len(train_loader):.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))

        # divide running loss by number of batches
        running_loss = running_loss / count
        # Allreduce, mean
        loss_avg = metric_average(running_loss)
        return {'loss': loss_avg}

    def test(self) -> dict:
        running_loss = torch.tensor(0.)
        count = torch.tensor(0.)
        if WITH_CUDA:
            running_loss = running_loss.cuda()
            count = count.cuda()
        self.model.eval()
        test_loader = self.data['test']['loader']
        with torch.no_grad():
            for data in test_loader:
                if self.device == 'gpu':
                    data = data.cuda()
                out = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
                loss = self.loss_fn(out, data.x)
                running_loss += loss.item()
                count += 1
            running_loss = running_loss / count
            loss_avg = metric_average(running_loss)
        return {'loss': loss_avg}


def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)


def train_mnist(cfg: DictConfig):
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []
    for epoch in range(trainer.epoch_start, cfg.epochs + 1):
        # ~~~~ Training step 
        t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        epoch_times.append(time.time() - t0)

        # ~~~~ Validation step
        test_metrics = trainer.test()
        if epoch % cfg.logfreq and RANK == 0:
            astr = f'[TEST] loss={test_metrics["loss"]:.4e}'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={train_metrics["loss"]:.4e}'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step 
        if epoch % cfg.ckptfreq and RANK == 0:
            astr = 'Checkpointing on root processor'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
            
            try:
                ckpt_path = cfg.ckpt_dir + trainer.model.get_save_header()
            except (AttributeError) as e:
                ckpt_path = cfg.ckpt_dir + 'checkpoint.pt'

            ckpt = {'epoch' : epoch, 
                    'model_state_dict' : trainer.model.state_dict(), 
                    'optimizer_state_dict' : trainer.optimizer.state_dict()} 
                    #'scheduler_state_dict' : scheduler.state_dict()}
            torch.save(ckpt, ckpt_path)
        dist.barrier()

    # ~~~~ # rstr = f'[{RANK}] ::'
    # ~~~~ # log.info(' '.join([
    # ~~~~ #     rstr,
    # ~~~~ #     f'Total training time: {time.time() - start} seconds'
    # ~~~~ # ]))
    #log.info(' '.join([
    #    rstr,
    #    f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    #]))


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    train_mnist(cfg)
    cleanup()


if __name__ == '__main__':
    main()
