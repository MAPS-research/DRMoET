#!/bin/bash
# Training launcher script for FLAME-MoE using local dataset.

# Setup environment variables for training and debugging

# Load site configuration (run from the repository root)
source scripts/config.sh

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# NCCL_NVLS_ENABLE=0 only needed for partial GPU allocations (<8 GPUs/node)
# With 8 GPUs/node, NVLS works correctly
export NCCL_NVLS_ENABLE=1

# Set up CUDA environment
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source "$CONDA_ROOT/bin/activate"
conda activate "$CONDA_ENV"

# Setup required arguments to Megatron-LM
source configs/model/flame-moe.sh
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 90,5,5
)

SAVE_ARGS=(
    --log-interval 5
    --log-throughput
    --save $SSD_WEIGHTS
    --save-interval $SAVE_INTERVAL
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --wandb-save-dir $SSD_WEIGHTS
    --wandb-project $WANDB_PROJECT
    --wandb-exp-name $SLURM_JOB_ID
    --tensorboard-dir $SSD_WEIGHTS
)

# Only add the --load argument if TRAIN_FROM_SCRATCH is not set to true
if [[ "$TRAIN_FROM_SCRATCH" != "true" ]]; then
    SAVE_ARGS+=(--load $SSD_WEIGHTS)
fi

EXTRA_ARGS=()
if [[ "$LOG_MOE_SIGNALS" = "true" ]]; then
    EXTRA_ARGS+=(--log-moe-signals)
fi
if [[ -n "$SEED" ]]; then
    EXTRA_ARGS+=(--seed "$SEED")
fi

# Start training with torchrun using host conda environment
cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" "${EXTRA_ARGS[@]}" 