#!/bin/bash
# Launch the evaluation, which traces the top-k router.

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

source configs/model/flame-moe.sh
source configs/train/flame-moe.sh

DATA_ARGS=(
    --seq-length 2048
    --data-path $(find $SSD_DATASET -type f -name '*.bin' -exec sh -c 'printf "1.0 %s " "${1%.bin}"' _ {} \; | sed 's/ $//')
    --split 95,0,5
)

SAVE_ARGS=(
    --test-mode
    --skip-train
    --load $TRAIN_WEIGHTS
    --finetune
    --no-load-optim
    --no-load-rng
    --eval-iters 25
)

cd Megatron-LM && torchrun "${TORCH_ARGS[@]}" pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"
