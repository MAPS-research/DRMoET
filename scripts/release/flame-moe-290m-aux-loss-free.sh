#!/bin/bash

# Load site configuration (run from the repository root)
source scripts/config.sh

export NUM_LAYERS=9
export HIDDEN_SIZE=1024
export FFN_HIDDEN_SIZE=5472
export MOE_FFN_HIDDEN_SIZE=704
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=32
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export TRAIN_ITERS=16000
# export SAVE_INTERVAL=540
export SAVE_INTERVAL=1600
export EVAL_INTERVAL=1600
export TRAIN_FROM_SCRATCH=false # Set to true to train from scratch

# Calculate model size identifier for weights directory
# This creates a unique identifier based on model architecture
export MODEL_SIZE_ID="290m-aux-loss-free-bias-update-rate-1e-2"  # Based on the script name and model parameters
export LOG_MOE_SIGNALS=true
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

# Calculate world size from SLURM configuration (4 nodes * 4 gpus_per_node)
WORLD_SIZE=16
# Calculate data parallel size (INCLUDING expert_model_parallel_size in denominator)
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (1 * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)))
# Calculate current effective batch size
CURRENT_EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
# Original effective batch size (4 nodes × 8 GPUs, data_parallel_size=4, micro_batch_size=8)
ORIGINAL_EFFECTIVE_BATCH_SIZE=32
# Calculate proportional eval_iters
export EVAL_ITERS=$((100 * CURRENT_EFFECTIVE_BATCH_SIZE / ORIGINAL_EFFECTIVE_BATCH_SIZE))

sbatch --job-name=flame-moe-290m-aux-loss-free --nodes=4 --gres=gpu:4 --mem=1024G scripts/training/flame-moe_local_aux_loss_free.slurm
