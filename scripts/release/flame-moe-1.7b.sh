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

# Calculate model size identifier for weights directory
export MODEL_SIZE_ID="1.7b_32k_${NODES}x${GPUS_PER_NODE}"
export LOG_MOE_SIGNALS=true
export TRAIN_FROM_SCRATCH=${TRAIN_FROM_SCRATCH:-true}
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_WEIGHTS="$SSD_MOUNT/weights/$MODEL_SIZE_ID"

# Calculate EVAL_ITERS automatically based on effective batch size
# Formula: eval_iters = 100 * (current_effective_batch_size / original_effective_batch_size)
# Original: 4 nodes × 8 GPUs = 32 GPUs, DATA_PARALLEL_SIZE=4, effective_batch_size=8*4=32, eval_iters=100 (default)
#
# Automatic calculation:
# - World size = nodes * gpus_per_node (from SLURM configuration)
# - Data parallel size = world_size / (tensor_model_parallel_size × expert_model_parallel_size × pipeline_model_parallel_size)
# - Current effective batch size = micro_batch_size × data_parallel_size
# - Original effective batch size = 8 × 4 = 32
# - EVAL_ITERS = 100 * (current_effective_batch_size / original_effective_batch_size)

# Calculate world size from SLURM configuration (2 nodes * 8 gpus_per_node)
WORLD_SIZE=16
# Calculate data parallel size (INCLUDING expert_model_parallel_size in denominator)
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (1 * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)))
# Calculate current effective batch size
CURRENT_EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
# Original effective batch size (4 nodes × 8 GPUs, data_parallel_size=4, micro_batch_size=8)
ORIGINAL_EFFECTIVE_BATCH_SIZE=32
# Calculate proportional eval_iters (current_effective=16, original=32, so 50)
export EVAL_ITERS=50

CPUS=$((GPUS_PER_NODE * 16))
MEM_PER_GPU_GB=250
MEM=$((GPUS_PER_NODE * MEM_PER_GPU_GB))G
sbatch --job-name=flame-moe-1.7b-${NODES}x${GPUS_PER_NODE} --nodes=$NODES --gres=gpu:$GPUS_PER_NODE --mem=$MEM --cpus-per-task=$CPUS scripts/training/flame-moe_local.slurm
