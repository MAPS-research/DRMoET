#!/bin/bash

# Load site configuration (run from the repository root)
source scripts/config.sh

export NUM_LAYERS=18
export HIDDEN_SIZE=2048
export FFN_HIDDEN_SIZE=10944
export MOE_FFN_HIDDEN_SIZE=1408
export MOE_LAYER_FREQ="[0]*1+[1]*17"
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-16}
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
NODES=${NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TRAIN_ITERS=32000
export SAVE_INTERVAL=3200
export EVAL_INTERVAL=3200

export MODEL_SIZE_ID="1.7b_aux_loss_free_bias1e-2_${NODES}x${GPUS_PER_NODE}"
export LOG_MOE_SIGNALS=true
export TRAIN_FROM_SCRATCH=${TRAIN_FROM_SCRATCH:-true}
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_WEIGHTS="$SSD_MOUNT/weights/$MODEL_SIZE_ID"

export EVAL_ITERS=50
export NCCL_NVLS_ENABLE=0
export SLURM_CPU_BIND=none

CPUS=$((GPUS_PER_NODE * 16))
MEM_PER_GPU_GB=250
MEM=$((GPUS_PER_NODE * MEM_PER_GPU_GB))G
sbatch --job-name=flame-moe-1.7b-aux-loss-free-${NODES}x${GPUS_PER_NODE} --nodes=$NODES --gres=gpu:$GPUS_PER_NODE --mem=$MEM --cpus-per-task=$CPUS scripts/training/flame-moe_local_aux_loss_free.slurm
