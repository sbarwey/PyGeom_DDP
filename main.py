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

        # ~~~~ Init training and testing loss history 
        self.loss_hist_train = np.zeros(self.cfg.epochs)
        self.loss_hist_test = np.zeros(self.cfg.epochs)

        # ~~~~ Noise setup
        self.noise_dist = []
        if self.cfg.use_noise:
            mu = 0.0 
            std = 1e-2 
            self.noise_dist = tdist.Normal(torch.tensor([mu]), torch.tensor([std]))

        # ~~~~ Init datasets
        self.bounding_box = [0.0, 0.0, 0.0, 0.0]
        self.data = self.setup_data()

        # ~~~~ Init model and move to gpu 
        self.model = self.build_model()
        if self.device == 'gpu':
            self.model.cuda()

        # ~~~~ Set model and checkpoint savepaths:
        try:
            self.ckpt_path = cfg.ckpt_dir + self.model.get_save_header() + '.tar'
            self.model_path = cfg.model_dir + self.model.get_save_header() + '.tar'
        except (AttributeError) as e:
            self.ckpt_path = cfg.ckpt_dir + 'checkpoint.tar'
            self.model_path = cfg.model_dir + 'model.tar'

        # ~~~~ Load model parameters if we are restarting from checkpoint
        self.epoch_start = 1
        self.training_iter = 0
        if self.cfg.restart:
            ckpt = torch.load(self.ckpt_path)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.epoch_start = ckpt['epoch'] + 1
            self.training_iter = ckpt['training_iter']
            self.loss_hist_train = ckpt['loss_hist_train']
            self.loss_hist_test = ckpt['loss_hist_test']

        # ~~~~ Wrap model in DDP
        if WITH_DDP and SIZE > 1:
            self.model = DDP(self.model)

        # ~~~~ Set loss function 
        self.loss_fn = nn.MSELoss()

        # ~~~~ Set optimizer 
        self.optimizer = self.build_optimizer(self.model)

        # ~~~~ Set scheduler 
        self.scheduler = self.build_scheduler(self.optimizer)

        # ~~~~ Load optimizer+scheduler parameters if we are restarting from checkpoint
        if self.cfg.restart:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if RANK == 0: 
                astr = 'RESTARTING FROM CHECKPOINT -- STATE AT EPOCH %d/%d' %(self.epoch_start-1, self.cfg.epochs)
                sepstr = '-' * len(astr)
                log.info(sepstr)
                log.info(astr)
                log.info(sepstr)

        # if WITH_CUDA:
        #    self.loss_fn = self.loss_fn.cuda()

    def build_model(self) -> nn.Module:
        
        bbox = [tnsr.item() for tnsr in self.bounding_box]
        model = gnn.Multiscale_MessagePassing(
                in_channels_node = 2,
                in_channels_edge = 3,
                hidden_channels = 128,
                n_mlp_encode = 3,
                n_mlp_mp = 2,
                n_mp_down = [8], # [4,4,4], #[2,2]
                n_mp_up = [], #[4,4], #[2]
                n_repeat_mp_up = 1,
                lengthscales = [], #[0.01, 0.02], #[0.5]
                bounding_box = bbox,
                act = F.elu,
                interpolation_mode = 'knn',
                name = 'test')
        
        #model = gnn.MP_GNN(in_channels_node=1,
        #                       in_channels_edge=3,
        #                       hidden_channels=8,
        #                       out_channels=1,
        #                       n_mlp_encode=3,
        #                       n_mlp_mp=2,
        #                       n_mp=2,
        #                       act=F.elu)
        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        # DDP: scale learning rate by the number of GPUs
        optimizer = optim.Adam(model.parameters(),
                               lr=SIZE * self.cfg.lr_init)
        return optimizer

    def build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                patience=5, threshold=0.0001, threshold_mode='rel',
                                cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)
        return scheduler

    def setup_torch(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

    def get_rollout_steps(self) -> int:
        if self.training_iter >= 0:
            L = self.cfg.rollout_steps
        return L

    def setup_data(self):
        kwargs = {}
        #if self.device == 'gpu':
        #    kwargs = {'num_workers': 1, 'pin_memory': True}

        device_for_loading = 'cpu'

        # ~~~~ MNIST
        # train_dataset, test_dataset = umnist.get_mnist_dataset(
        #     self.cfg.path_to_vtk, self.cfg.path_to_ei, self.cfg.path_to_ea, 
        #     self.cfg.path_to_pos, device_for_loading
        # )

        # ~~~~ BFS
        filenames = ['Backward_Facing_Step_Cropped_Re_26214.vtk', 
                     'Backward_Facing_Step_Cropped_Re_29307.vtk', 
                     'Backward_Facing_Step_Cropped_Re_39076.vtk', 
                     'Backward_Facing_Step_Cropped_Re_45589.vtk'] 
        train_dataset = []
        test_dataset = []
        for f in filenames: 
            path_to_vtk = self.cfg.data_dir + '/BACKWARD_FACING_STEP/%s' %(f)
            train_dataset_temp, test_dataset_temp = bfs.get_pygeom_dataset_cell_data_radius(
                path_to_vtk, self.cfg.path_to_ei, self.cfg.path_to_ea,
                self.cfg.path_to_pos, device_for_loading, 
                time_lag = self.cfg.rollout_steps,
                features_to_keep = [1,2], 
                fraction_valid = 0.1, 
                multiple_cases = False)
            train_dataset = train_dataset + train_dataset_temp
            test_dataset = test_dataset + test_dataset_temp

        self.bounding_box = train_dataset[0].bounding_box

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
        rollout_length = self.get_rollout_steps()
        loss = torch.tensor([0.0])
        loss_scale = torch.tensor([1.0/rollout_length])

        if WITH_CUDA:
            data.x = data.x.cuda()
            data.edge_index = data.edge_index.cuda()
            data.edge_attr = data.edge_attr.cuda()
            data.pos = data.pos.cuda()
            data.batch = data.batch.cuda()
            loss = loss.cuda()
            loss_scale = loss_scale.cuda()
        
        self.optimizer.zero_grad()

        ## Single prediction:
        #out = self.model(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
        #loss = self.loss_fn(out, data.x)

        # Rollout prediction: 
        x_new = data.x
        for t in range(rollout_length):
            if self.cfg.use_noise and t == 0:
                noise = self.noise_dist.sample((data.x.shape[0],))
                if WITH_CUDA:
                    noise = noise.cuda()
                x_old = torch.clone(x_new) + noise
            else:
                x_old = torch.clone(x_new)

            x_src = self.model(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
            x_new = x_old + x_src

            # Accumulate loss 
            target = data.y[t]
            if WITH_CUDA:
                target = target.cuda()
            loss += loss_scale * self.loss_fn(x_new, target)

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
            #print('Rank %d, bid %d, data:' %(RANK, bidx), data.y[1].shape)
            loss = self.train_step(data)
            running_loss += loss.item()
            count += 1 # accumulate current batch count 
            self.training_iter += 1 # accumulate total training iteration
            
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
                rollout_length = self.get_rollout_steps()
                loss = torch.tensor([0.0])
                loss_scale = torch.tensor([1.0/rollout_length])

                if WITH_CUDA:
                    data.x = data.x.cuda()
                    data.edge_index = data.edge_index.cuda()
                    data.edge_attr = data.edge_attr.cuda()
                    data.pos = data.pos.cuda()
                    data.batch = data.batch.cuda()
                    loss = loss.cuda()
                    loss_scale = loss_scale.cuda()
                
                # Rollout prediction: 
                x_new = data.x
                for t in range(rollout_length):
                    x_old = torch.clone(x_new)
                    x_src = self.model(x_old, data.edge_index, data.edge_attr, data.pos, data.batch)
                    x_new = x_old + x_src

                    # Accumulate loss 
                    target = data.y[t]
                    if WITH_CUDA:
                        target = target.cuda()
                    loss += loss_scale * self.loss_fn(x_new, target)

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

def train(cfg: DictConfig):
    start = time.time()
    trainer = Trainer(cfg)
    epoch_times = []
    for epoch in range(trainer.epoch_start, cfg.epochs+1):
        # ~~~~ Training step 
        t0 = time.time()
        train_metrics = trainer.train_epoch(epoch)
        trainer.loss_hist_train[epoch-1] = train_metrics["loss"]
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # ~~~~ Validation step
        test_metrics = trainer.test()
        trainer.loss_hist_test[epoch-1] = test_metrics["loss"]
        if RANK == 0:
            astr = f'[TEST] loss={test_metrics["loss"]:.4e}'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={train_metrics["loss"]:.4e}', 
                f'epoch_time={epoch_time:.4g} sec'
            ])
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)


        # ~~~~ Step scheduler based on validation loss
        trainer.scheduler.step(test_metrics["loss"])

        # ~~~~ Checkpointing step 
        if epoch % cfg.ckptfreq == 0 and RANK == 0:
            astr = 'Checkpointing on root processor, epoch = %d' %(epoch)
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)

            if not os.path.exists(cfg.ckpt_dir):
                os.makedirs(cfg.ckpt_dir)
           
            if WITH_DDP and SIZE > 1:
                ckpt = {'epoch' : epoch, 
                        'training_iter' : trainer.training_iter,
                        'model_state_dict' : trainer.model.module.state_dict(), 
                        'optimizer_state_dict' : trainer.optimizer.state_dict(), 
                        'scheduler_state_dict' : trainer.scheduler.state_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test}
            else:
                ckpt = {'epoch' : epoch, 
                        'training_iter' : trainer.training_iter,
                        'model_state_dict' : trainer.model.state_dict(), 
                        'optimizer_state_dict' : trainer.optimizer.state_dict(), 
                        'scheduler_state_dict' : trainer.scheduler.state_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test}

            torch.save(ckpt, trainer.ckpt_path)
        dist.barrier()

    rstr = f'[{RANK}] ::'
    log.info(' '.join([
        rstr,
        f'Total training time: {time.time() - start} seconds'
    ]))
    #log.info(' '.join([
    #    rstr,
    #    f'Average time per epoch in the last 5: {np.mean(epoch_times[-5])}'
    #]))

    if RANK == 0:
        if WITH_CUDA:  
            trainer.model.to('cpu')
        if not os.path.exists(cfg.model_dir):
            os.makedirs(cfg.model_dir)
        if WITH_DDP and SIZE > 1:
            save_dict = {
                        'state_dict' : trainer.model.module.state_dict(), 
                        'input_dict' : trainer.model.module.input_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'training_iter' : trainer.training_iter
                        }
        else:
            save_dict = {   
                        'state_dict' : trainer.model.state_dict(), 
                        'input_dict' : trainer.model.input_dict(),
                        'loss_hist_train' : trainer.loss_hist_train,
                        'loss_hist_test' : trainer.loss_hist_test,
                        'training_iter' : trainer.training_iter
                        }

        torch.save(save_dict, trainer.model_path)

@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print('Rank %d, local rank %d, which has device %s. Sees %d devices.' %(RANK,int(LOCAL_RANK),DEVICE,torch.cuda.device_count()))
    train(cfg)
    cleanup()

if __name__ == '__main__':
    main()
