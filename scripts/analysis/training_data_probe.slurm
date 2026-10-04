#!/bin/bash
# Capture per-expert routing probability and loss on training data.
# Uses the same data pipeline and capture mechanism as the utilization analysis,
# but also computes per-expert losses via --log-moe-signals.

#SBATCH --job-name=train-data-probe
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Usage:
#   MODEL_SIZE_ID=290m-seed3407-32e ITER=16000 sbatch scripts/analysis/training_data_probe.slurm

# Load site configuration (run from the repository root)
source scripts/config.sh

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64

source "$CONDA_ROOT/bin/activate"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=16
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export MEGATRON_USE_TORCH_COMPILE=1

if [ -z "${MODEL_SIZE_ID:-}" ]; then
    echo "Error: MODEL_SIZE_ID must be set"
    exit 1
fi

# Model architecture
case "$MODEL_SIZE_ID" in
    290m*)
        export NUM_LAYERS=9; export HIDDEN_SIZE=1024; export FFN_HIDDEN_SIZE=5472
        export MOE_FFN_HIDDEN_SIZE=704; export MOE_LAYER_FREQ="[0]*1+[1]*8"
        export NUM_EXPERTS=32
        ;;
    1.7b*)
        export NUM_LAYERS=18; export HIDDEN_SIZE=2048; export FFN_HIDDEN_SIZE=10944
        export MOE_FFN_HIDDEN_SIZE=1408; export MOE_LAYER_FREQ="[0]*1+[1]*17"
        export NUM_EXPERTS=64
        ;;
    *)
        echo "Error: Unknown MODEL_SIZE_ID: $MODEL_SIZE_ID"; exit 1
        ;;
esac

export MICRO_BATCH_SIZE=4
export TRAIN_ITERS=1
export PIPELINE_MODEL_PARALLEL_SIZE=1
export EXPERT_MODEL_PARALLEL_SIZE=2
export MOE_ROUTER_TOPK=6
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="localhost:$((29500 + SLURM_JOB_ID % 1000))"

export SSD_MOUNT="${SSD_MOUNT:-$PROJECT_ROOT}"
export SSD_DATASET="$SSD_MOUNT/dataset/tokenized/EleutherAI/pythia-12b"
export SSD_WEIGHTS="$SSD_MOUNT/weights"

CHECKPOINT_DIR="$SSD_WEIGHTS/$MODEL_SIZE_ID"
ITER=${ITER:-$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo "16000")}
echo "$ITER" > "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt"

OUTPUT_DIR="$SSD_MOUNT/results/training-data-probe/$MODEL_SIZE_ID/iter_$ITER"
mkdir -p "$OUTPUT_DIR"

# SSD_WEIGHTS is used for both checkpoint loading and expert_stats saving
# Keep it for loading, but we'll override after building args

# Set EACT_SAVE so routing traces are saved
export EACT_SAVE="$OUTPUT_DIR/actives"
export TIDS_SAVE="$OUTPUT_DIR/samples"

echo "=========================================="
echo "Training Data Probe: $MODEL_SIZE_ID iter=$ITER"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

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
    --load "$CHECKPOINT_DIR"
    --finetune
    --no-load-optim
    --no-load-rng
    --eval-iters 25
    --log-moe-signals
    --save-interval 1000
    --log-interval 1
)

# Override SSD_WEIGHTS so expert_stats saves to model-specific output dir
# (must be after configs are sourced but before torchrun, since _save_expert_stats_if_needed reads it at runtime)
export SSD_WEIGHTS="$OUTPUT_DIR"

cd Megatron-LM && torchrun --nproc_per_node=2 --master_port $((29500 + SLURM_JOB_ID % 1000)) \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" "${INFRA_ARGS[@]}" "${TRAIN_ARGS[@]}" "${DATA_ARGS[@]}" "${SAVE_ARGS[@]}"

echo "Training data probe complete. Results in $OUTPUT_DIR"
