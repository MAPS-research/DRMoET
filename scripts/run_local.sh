#!/bin/bash
# Load site configuration (run from the repository root)
source scripts/config.sh

set -euo pipefail

# Run a FLAME-MoE release script locally without SLURM.
# Usage: bash scripts/run_local.sh scripts/release/flame-moe-290m-aux-loss-free.sh

RELEASE_SCRIPT="${1:?Usage: bash scripts/run_local.sh <release-script>}"

# Set up logging (mirrors SLURM log structure)
LOG_DIR="logs/$(basename "$RELEASE_SCRIPT" .sh)"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${TIMESTAMP}.log"

# Source all exports from the release script (skip the sbatch line)
source <(grep -v '^sbatch ' "$RELEASE_SCRIPT")

# Detect GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Simulate SLURM environment for single-node
export SLURM_NNODES=1
export SLURM_NODEID=0
export SLURM_GPUS_ON_NODE=$NUM_GPUS
export SLURM_JOB_ID=${SLURM_JOB_ID:-$$}
export SLURM_JOB_NAME=$(basename "$RELEASE_SCRIPT" .sh)
export RDZV_BACKEND="c10d"
export RDZV_ENDPOINT="localhost:8000"

# Dataset and weights
export SSD_DATASET="$SSD_MOUNT/dataset/tokenized/EleutherAI/pythia-12b"

# Recalculate EVAL_ITERS for actual GPU count
WORLD_SIZE=$NUM_GPUS
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (1 * EXPERT_MODEL_PARALLEL_SIZE * PIPELINE_MODEL_PARALLEL_SIZE)))
CURRENT_EFFECTIVE_BATCH_SIZE=$((MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE))
ORIGINAL_EFFECTIVE_BATCH_SIZE=32
export EVAL_ITERS=$((100 * CURRENT_EFFECTIVE_BATCH_SIZE / ORIGINAL_EFFECTIVE_BATCH_SIZE))

# Weights & Biases
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-pretrain_drmoet}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-$SLURM_JOB_NAME}"
export WANDB_NAME="${WANDB_NAME:-$SLURM_JOB_ID}"

mkdir -p "$SSD_WEIGHTS"

# Extract the training module from the sbatch -> srun chain
SLURM_SCRIPT=$(grep '^sbatch ' "$RELEASE_SCRIPT" | grep -oE 'scripts/[^ ]+\.slurm')
TRAINING_MODULE=$(grep 'srun' "$SLURM_SCRIPT" | grep -oE 'scripts/[^ ]+\.sh')

echo "=== FLAME-MoE Local Training ==="
echo "Release:    $RELEASE_SCRIPT"
echo "Module:     $TRAINING_MODULE"
echo "GPUs:       $NUM_GPUS"
echo "DP:         $DATA_PARALLEL_SIZE"
echo "EP:         $EXPERT_MODEL_PARALLEL_SIZE"
echo "EVAL_ITERS: $EVAL_ITERS"
echo "Weights:    $SSD_WEIGHTS"
echo "Log:        $LOG_FILE"
echo "================================"

exec bash "$TRAINING_MODULE" > >(tee "$LOG_FILE") 2>&1
