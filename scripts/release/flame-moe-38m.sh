#!/bin/bash

# Load site configuration (run from the repository root)
source scripts/config.sh

export NUM_LAYERS=9
export HIDDEN_SIZE=256
export FFN_HIDDEN_SIZE=1368
export MOE_FFN_HIDDEN_SIZE=176
export MOE_LAYER_FREQ="[0]*1+[1]*8"
export MICRO_BATCH_SIZE=32
# Original: MICRO_BATCH_SIZE=32, EXPERT_MODEL_PARALLEL_SIZE=8 (8 GPUs)
# New: MICRO_BATCH_SIZE=32, EXPERT_MODEL_PARALLEL_SIZE=2 (2 GPUs)
# Keep micro batch size the same to maintain effective batch size
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export TRAIN_ITERS=2121
export SAVE_INTERVAL=212
export EVAL_INTERVAL=212

# Calculate model size identifier for weights directory
# This creates a unique identifier based on model architecture
export MODEL_SIZE_ID="38m"  # Based on the script name and model parameters
export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_WEIGHTS="$SSD_MOUNT/weights/$MODEL_SIZE_ID"

# Calculate EVAL_ITERS automatically based on effective batch size
# Formula: eval_iters = 100 * (current_effective_batch_size / original_effective_batch_size)
# Original: 4 nodes × 8 GPUs = 32 GPUs, DATA_PARALLEL_SIZE=4, effective_batch_size=32*4=128, eval_iters=100 (default)
# 
# Automatic calculation:
# - World size = nodes * gpus_per_node (from SLURM configuration)
# - Data parallel size = world_size / (tensor_model_parallel_size × expert_model_parallel_size × pipeline_model_parallel_size)
# - Current effective batch size = micro_batch_size × data_parallel_size
# - Original effective batch size = 32 × 4 = 128
# - EVAL_ITERS = 100 * (current_effective_batch_size / original_effective_batch_size)

# Calculate world size from SLURM configuration (nodes * gpus_per_node)
WORLD_SIZE=2  # 2 nodes * 1 GPU per node
# Calculate data parallel size (INCLUDING expert_model_parallel_size in denominator)
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (1 * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)))
# Calculate current effective batch size
CURRENT_EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
# Original effective batch size (4 nodes × 8 GPUs, data_parallel_size=4)
ORIGINAL_EFFECTIVE_BATCH_SIZE=128
# Calculate proportional eval_iters
export EVAL_ITERS=$((100 * CURRENT_EFFECTIVE_BATCH_SIZE / ORIGINAL_EFFECTIVE_BATCH_SIZE))

sbatch --job-name=flame-moe-38m --nodes=2 scripts/training/flame-moe_local.slurm
