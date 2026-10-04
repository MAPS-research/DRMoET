#!/bin/bash
# Training launcher script for FLAME-MoE using local dataset with full DRO support.

# Setup environment variables for training and debugging

# Load site configuration (run from the repository root)
source scripts/config.sh

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export TORCH_NCCL_TRACE_BUFFER_SIZE=8
export TORCH_NCCL_DUMP_ON_TIMEOUT=1

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

DRO_ARGS=()
if [[ -n "$USE_DRO" ]]; then
    DRO_ARGS+=(--use-dro)
    DRO_ARGS+=(--dro-beta "$DRO_BETA")
    DRO_ARGS+=(--dro-eta "$DRO_ETA")
    DRO_ARGS+=(--dro-alpha "$DRO_ALPHA")
fi

if [[ "$LOG_DRO_SIGNALS" = "true" ]]; then
    DRO_ARGS+=(--log-dro-signals)
fi

if [[ -n "$DRO_ROUTER_GRAD_CLIP" ]]; then
    DRO_ARGS+=(--dro-router-grad-clip "$DRO_ROUTER_GRAD_CLIP")
fi

if [[ -n "$DRO_MU_UPDATE_CLIP" ]]; then
    DRO_ARGS+=(--dro-mu-update-clip "$DRO_MU_UPDATE_CLIP")
fi

if [[ "$USE_QK_NORM" = "true" ]]; then
    MODEL_ARGS+=(--qk-layernorm)
fi

# Start training with torchrun using the full DRO pretrain script
cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt_dro.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}" "${DRO_ARGS[@]}" 
