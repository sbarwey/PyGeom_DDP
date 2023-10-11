#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=4:00:00
##PBS -l walltime=1:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
##PBS -q debug
#PBS -A datascience
#PBS -N GNN_DDP


# Change to working directory
cd ${PBS_O_WORKDIR}

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"


# Load modules: 
source /lus/eagle/projects/datascience/sbarwey/codes/ml/pytorch_geometric/module_config

# Get number of ranks 
NUM_NODES=$(wc -l < "${PBS_NODEFILE}")

# Get number of GPUs per node
NGPUS_PER_NODE=$(nvidia-smi -L | wc -l)

# Get total number of GPUs 
NGPUS="$((${NUM_NODES}*${NGPUS_PER_NODE}))"

# Print 
echo $NUM_NODES $NGPUS_PER_NODE $NGPUS

# run 
mpiexec \
	--verbose \
	--envall \
	-n $NGPUS \
	--ppn $NGPUS_PER_NODE \
	--hostfile="${PBS_NODEFILE}" \
    --cpu-bind none \
	./set_affinity_gpu_polaris.sh python3 main.py seed=65 use_noise=False baseline_modelpath=/lus/eagle/projects/datascience/sbarwey/codes/ml/DDP_PyGeom/saved_models/big_data/dt_gnn_1em4/NO_NOISE_NO_RADIUS_LR_1em5_topk_unet_rollout_1_seed_82_down_topk_2_up_topk_factor_4_hc_128_down_enc_2_2_2_up_enc_2_2_down_dec_2_2_2_up_dec_2_2_param_sharing_0.tar
